#!/usr/bin/env python
"""Standalone K-MIMIC preprocessing pipeline for meds-torch.

Replicates the MEDS-transforms triplet pipeline without needing
meds_torch.utils.tensorize (missing in this version) or
meds_torch.utils.get_all_measurements (also missing).

Usage:
    python experiments/lane_a/preprocess_kmimic.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

MEDS_DIR = Path("data/output")
OUT_DIR = Path("data/kmimic_triplet_tensors")
SPLITS = ["train", "tuning", "held_out"]

MIN_SUBJECTS_PER_CODE = 10
MIN_EVENTS_PER_SUBJECT = 10
OUTLIER_STD_CUTOFF = 4.5
SECONDS_PER_DAY = 86_400.0


# ---------------------------------------------------------------------------
# Step 1 – load raw MEDS data
# ---------------------------------------------------------------------------

def load_split(split: str) -> pl.DataFrame:
    files = list((MEDS_DIR / "data" / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {MEDS_DIR}/data/{split}/")
    return pl.concat([pl.read_parquet(f) for f in sorted(files)])


def load_all() -> dict[str, pl.DataFrame]:
    return {s: load_split(s) for s in SPLITS}


# ---------------------------------------------------------------------------
# Step 2 – compute train-set code statistics
# ---------------------------------------------------------------------------

def compute_code_stats(train: pl.DataFrame) -> pl.DataFrame:
    numeric = train.filter(pl.col("numeric_value").is_not_null())
    stats = (
        train.group_by("code")
        .agg(
            pl.n_unique("subject_id").alias("n_subjects"),
            pl.len().alias("n_occurrences"),
        )
        .join(
            numeric.group_by("code").agg(
                pl.col("numeric_value").sum().alias("values_sum"),
                (pl.col("numeric_value") ** 2).sum().alias("values_sum_sqd"),
                pl.col("numeric_value").len().alias("values_n"),
            ),
            on="code",
            how="left",
        )
    )
    # Compute mean and std from sum / sum_sqd
    stats = stats.with_columns(
        (pl.col("values_sum") / pl.col("values_n")).alias("values_mean"),
    ).with_columns(
        (
            (pl.col("values_sum_sqd") / pl.col("values_n") - pl.col("values_mean") ** 2).clip(lower_bound=0)
            .sqrt()
        ).alias("values_std"),
    )
    return stats


# ---------------------------------------------------------------------------
# Step 3 – filter codes and subjects
# ---------------------------------------------------------------------------

def filter_codes(dfs: dict, stats: pl.DataFrame) -> dict:
    keep_codes = set(
        stats.filter(pl.col("n_subjects") >= MIN_SUBJECTS_PER_CODE)["code"].to_list()
    )
    return {s: df.filter(pl.col("code").is_in(keep_codes)) for s, df in dfs.items()}


def filter_subjects(dfs: dict) -> dict:
    out = {}
    train_valid = (
        dfs["train"]
        .group_by("subject_id")
        .agg(pl.len().alias("n_events"))
        .filter(pl.col("n_events") >= MIN_EVENTS_PER_SUBJECT)["subject_id"]
        .to_list()
    )
    train_valid_set = set(train_valid)
    for split, df in dfs.items():
        if split == "train":
            out[split] = df.filter(pl.col("subject_id").is_in(train_valid_set))
        else:
            out[split] = df
    return out


# ---------------------------------------------------------------------------
# Step 4 – occlude outliers (set to null / NaN)
# ---------------------------------------------------------------------------

def occlude_outliers(dfs: dict, stats: pl.DataFrame) -> dict:
    stats_with_bounds = stats.with_columns(
        (pl.col("values_mean") - OUTLIER_STD_CUTOFF * pl.col("values_std")).alias("lower"),
        (pl.col("values_mean") + OUTLIER_STD_CUTOFF * pl.col("values_std")).alias("upper"),
    ).select(["code", "lower", "upper"])

    out = {}
    for split, df in dfs.items():
        merged = df.join(stats_with_bounds, on="code", how="left")
        merged = merged.with_columns(
            pl.when(
                pl.col("numeric_value").is_not_null()
                & (
                    (pl.col("numeric_value") < pl.col("lower"))
                    | (pl.col("numeric_value") > pl.col("upper"))
                )
            )
            .then(None)
            .otherwise(pl.col("numeric_value"))
            .alias("numeric_value")
        ).drop(["lower", "upper"])
        out[split] = merged
    return out


# ---------------------------------------------------------------------------
# Step 5 – fit vocabulary (lexicographic code → index)
# ---------------------------------------------------------------------------

def fit_vocabulary(train: pl.DataFrame) -> dict[str, int]:
    codes = sorted(train["code"].unique().to_list())
    return {c: i for i, c in enumerate(codes)}


# ---------------------------------------------------------------------------
# Step 6 – normalize numeric values (z-score from train stats)
# ---------------------------------------------------------------------------

def normalize(dfs: dict, stats: pl.DataFrame, vocab: dict[str, int]) -> dict:
    # Map code strings → int indices and apply z-score normalisation
    code_to_idx = pl.DataFrame(
        {"code": list(vocab.keys()), "code_idx": list(vocab.values())}
    ).with_columns(pl.col("code_idx").cast(pl.Int64))

    norm_params = stats.select(["code", "values_mean", "values_std"])

    out = {}
    for split, df in dfs.items():
        d = (
            df.join(code_to_idx, on="code", how="left")
            .join(norm_params, on="code", how="left")
        )
        d = d.with_columns(
            pl.when(
                pl.col("numeric_value").is_not_null()
                & pl.col("values_std").is_not_null()
                & (pl.col("values_std") > 0)
            )
            .then(
                (pl.col("numeric_value") - pl.col("values_mean")) / pl.col("values_std")
            )
            .otherwise(pl.col("numeric_value"))
            .cast(pl.Float32)
            .alias("numeric_value")
        ).drop(["values_mean", "values_std"])
        out[split] = d
    return out


# ---------------------------------------------------------------------------
# Step 7 – tokenize: build schema files and event-sequence parquets
# ---------------------------------------------------------------------------

def tokenize(dfs: dict, out_dir: Path):
    schema_root = out_dir / "tokenization" / "schemas"
    event_root = out_dir / "tokenization" / "event_seqs"

    for split, df in dfs.items():
        schema_dir = schema_root / split
        event_dir = event_root / split
        schema_dir.mkdir(parents=True, exist_ok=True)
        event_dir.mkdir(parents=True, exist_ok=True)

        static = df.filter(pl.col("time").is_null())
        dynamic = df.filter(pl.col("time").is_not_null()).sort(["subject_id", "time"])

        # Schema: static codes + start_time + list of unique event times (matching meds-torch format)
        static_by_subj = (
            static
            .group_by("subject_id", maintain_order=True)
            .agg(
                pl.col("code_idx").alias("code"),
                pl.col("numeric_value").alias("numeric_value"),
            )
        )
        schema_by_subj = (
            dynamic
            .group_by("subject_id", maintain_order=True)
            .agg(
                pl.col("time").min().alias("start_time"),
                pl.col("time").unique(maintain_order=True).alias("time"),
            )
        )
        # All subjects must appear in schema (join full on subject_id)
        all_subjects = df.select("subject_id").unique()
        schema = (
            all_subjects
            .join(static_by_subj, on="subject_id", how="left")
            .join(schema_by_subj, on="subject_id", how="left")
        )
        schema.write_parquet(schema_dir / "0.parquet", use_pyarrow=True)

        # Event sequences per subject
        time_delta = (
            dynamic
            .with_columns(
                (
                    pl.col("time").diff().dt.total_seconds() / SECONDS_PER_DAY
                ).over("subject_id").cast(pl.Float32).alias("time_delta_days")
            )
        )

        event_seqs = (
            time_delta
            .group_by(["subject_id", "time"], maintain_order=True)
            .agg(
                pl.col("code_idx").alias("code"),
                pl.col("numeric_value").alias("numeric_value"),
                pl.col("time_delta_days").first().alias("time_delta_days"),
            )
            .group_by("subject_id", maintain_order=True)
            .agg(
                pl.col("time_delta_days"),
                pl.col("code"),
                pl.col("numeric_value"),
            )
        )
        event_seqs.write_parquet(event_dir / "0.parquet", use_pyarrow=True)


# ---------------------------------------------------------------------------
# Step 8 – tensorize: parquet event_seqs → .nrt files
# ---------------------------------------------------------------------------

def _to_float32_list(series: pl.Series) -> list[float]:
    return [float(x) if x is not None else float("nan") for x in series.to_list()]


def _to_nested_int(series: pl.Series) -> list[list[int]]:
    return [list(row) for row in series.to_list()]


def _to_nested_float(series: pl.Series) -> list[list[float]]:
    result = []
    for row in series.to_list():
        result.append([float(x) if x is not None else float("nan") for x in row])
    return result


def tensorize(out_dir: Path):
    event_root = out_dir / "tokenization" / "event_seqs"
    data_root = out_dir / "data"

    for split in SPLITS:
        event_dir = event_root / split
        data_dir = data_root / split
        data_dir.mkdir(parents=True, exist_ok=True)

        for fp in sorted(event_dir.glob("*.parquet")):
            shard_name = fp.stem
            df = pl.read_parquet(fp)

            all_subjects = []
            for row in df.iter_rows(named=True):
                t = _to_float32_list(pl.Series(row["time_delta_days"]))
                # first time_delta_days is NaN (no prior event)
                if t and not np.isnan(t[0]):
                    t[0] = float("nan")

                codes = _to_nested_int(pl.Series(row["code"]))
                values = _to_nested_float(pl.Series(row["numeric_value"]))

                subj = JointNestedRaggedTensorDict(
                    raw_tensors={
                        "time_delta_days": t,
                        "code": codes,
                        "numeric_value": values,
                    }
                )
                all_subjects.append(subj)

            if not all_subjects:
                continue

            combined = JointNestedRaggedTensorDict.vstack(all_subjects)
            nrt_fp = data_dir / f"{shard_name}.nrt"
            if not nrt_fp.exists():
                combined.save(nrt_fp)
            print(f"  Saved {nrt_fp} ({len(all_subjects)} subjects)")


# ---------------------------------------------------------------------------
# Step 9 – write codes.parquet metadata + task_info.json
# ---------------------------------------------------------------------------

def write_metadata(vocab: dict[str, int], stats: pl.DataFrame, out_dir: Path):
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # codes.parquet: maps each code to its index and stats
    codes_df = (
        pl.DataFrame(
            {
                "code": list(vocab.keys()),
                "code/vocab_index": [vocab[c] for c in vocab.keys()],
            }
        )
        .join(stats.select(["code", "n_subjects", "n_occurrences", "values_mean", "values_std"]), on="code", how="left")
    )
    codes_df.write_parquet(meta_dir / "codes.parquet", use_pyarrow=True)


def write_task_info(labels_path: Path, info_path: Path):
    df = pl.read_parquet(labels_path)
    n_pos = int(df["boolean_value"].sum())
    n_total = len(df)
    info = {
        "n_subjects": n_total,
        "n_positive": n_pos,
        "n_negative": n_total - n_pos,
        "positive_prevalence": n_pos / n_total if n_total > 0 else 0.0,
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(f"  Task info: {n_pos}/{n_total} positive ({info['positive_prevalence']:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== K-MIMIC Preprocessing Pipeline ===\n")

    print("Step 1: Loading raw MEDS data...")
    dfs = load_all()
    for s, df in dfs.items():
        print(f"  {s}: {len(df):,} events, {df['subject_id'].n_unique()} subjects")

    print("\nStep 2: Computing train code statistics...")
    stats = compute_code_stats(dfs["train"])
    print(f"  {len(stats)} codes with stats computed")

    print("\nStep 3: Filtering codes and subjects...")
    dfs = filter_codes(dfs, stats)
    dfs = filter_subjects(dfs)
    # Recompute stats on filtered train
    stats = compute_code_stats(dfs["train"])
    for s, df in dfs.items():
        print(f"  {s}: {len(df):,} events, {df['subject_id'].n_unique()} subjects")

    print("\nStep 4: Occluding outliers...")
    dfs = occlude_outliers(dfs, stats)

    print("\nStep 5: Building vocabulary...")
    vocab = fit_vocabulary(dfs["train"])
    print(f"  Vocabulary size: {len(vocab)}")

    print("\nStep 6: Normalizing...")
    dfs = normalize(dfs, stats, vocab)

    print("\nStep 7: Tokenizing...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenize(dfs, OUT_DIR)
    print(f"  Written to {OUT_DIR}/tokenization/")

    print("\nStep 8: Tensorizing...")
    tensorize(OUT_DIR)

    print("\nStep 9: Writing metadata...")
    write_metadata(vocab, stats, OUT_DIR)

    labels_path = Path("data/labels/inhospital_mortality_24h.parquet")
    info_path = Path("data/labels/inhospital_mortality_24h_info.json")
    if labels_path.exists():
        write_task_info(labels_path, info_path)
    else:
        print(f"  Warning: {labels_path} not found, skipping task info")

    print(f"\nDone. Output in {OUT_DIR}/")


if __name__ == "__main__":
    main()
