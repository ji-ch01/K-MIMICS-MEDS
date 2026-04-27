"""
Lane B — First-24h feature extraction from a MEDS dataset.

For each (subject_id, prediction_time) pair in the label file, extracts
harmonised summary features over the window [prediction_time - 24h, prediction_time].

Numeric concepts  : first, last, min, max, mean, count, missing_flag
Static features   : age (years at prediction_time), sex_male (0/1)
Event counts      : one feature per event family (count in window)
Fluid balance     : total_input, n_input_events, total_output, n_output_events

Concept↔code mappings are read from concepts.yaml (one entry per dataset).

Usage — K-MIMIC (three small split files):
    python experiments/lane_b/feature_extract.py \\
        --meds_dir      data/output \\
        --labels_path   data/labels/inhospital_mortality_24h \\
        --dataset       kmimic \\
        --concepts      experiments/concepts.yaml \\
        --output_dir    experiments/lane_b/features/kmimic

Usage — MIMIC-IV (292 train shards, labels are a single parquet file):
    python experiments/lane_b/feature_extract.py \\
        --meds_dir      data/MEDS_cohort \\
        --labels_path   data/MEDS_cohort/tasks/mortality/in_hospital/first_24h.parquet \\
        --splits_path   data/MEDS_cohort/metadata/subject_splits.parquet \\
        --dataset       mimic \\
        --concepts      experiments/concepts.yaml \\
        --output_dir    experiments/lane_b/features/mimic \\
        --splits        train tuning held_out
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GAP = pd.Timedelta(hours=24)
STATS = ["first", "last", "min", "max", "mean", "count", "missing"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_concepts(concepts_path: str, dataset: str) -> dict:
    with open(concepts_path) as f:
        all_cfg = yaml.safe_load(f)
    if dataset not in all_cfg:
        raise KeyError(f"Dataset '{dataset}' not found in {concepts_path}. "
                       f"Available: {list(all_cfg.keys())}")
    return all_cfg[dataset]


def code_matches(code_series: pd.Series, prefixes: list[str]) -> pd.Series:
    """True where code equals any prefix or starts with '{prefix}//'."""
    mask = pd.Series(False, index=code_series.index)
    for p in prefixes:
        mask |= (code_series == p) | code_series.str.startswith(p + "//", na=False)
    return mask


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels_kmimic(labels_path: str) -> pd.DataFrame:
    """
    Load K-MIMIC labels from MEDS-DEV split directories.
    Each split dir contains 0.parquet with [subject_id, prediction_time, boolean_value].
    Returns a DataFrame with an added 'split' column.
    """
    path = Path(labels_path)
    frames = []
    for split in ["train", "tuning", "held_out"]:
        fp = path / split / "0.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            df["split"] = split
            frames.append(df)
    labels = pd.concat(frames, ignore_index=True)
    labels["prediction_time"] = pd.to_datetime(labels["prediction_time"])
    return labels[["subject_id", "prediction_time", "boolean_value", "split"]]


def load_labels_mimic(labels_path: str, splits_path: str) -> pd.DataFrame:
    """
    Load MIMIC labels from a single parquet file and join with subject_splits.
    """
    labels = pd.read_parquet(labels_path)
    labels["prediction_time"] = pd.to_datetime(labels["prediction_time"])
    splits = pd.read_parquet(splits_path)[["subject_id", "split"]]
    labels = labels.merge(splits, on="subject_id", how="inner")
    return labels[["subject_id", "prediction_time", "boolean_value", "split"]]


# ---------------------------------------------------------------------------
# Per-subject feature computation
# ---------------------------------------------------------------------------

def _f2c(series: pd.Series, f_codes_mask: pd.Series) -> pd.Series:
    """Convert °F → °C for rows where f_codes_mask is True."""
    out = series.copy()
    out[f_codes_mask] = (series[f_codes_mask] - 32.0) * 5.0 / 9.0
    return out


def compute_features(
    events: pd.DataFrame,
    prediction_time: pd.Timestamp,
    cfg: dict,
) -> dict:
    """
    Compute all features for one (subject, prediction_time).

    events : all MEDS rows for this subject (any time, including NaT)
    """
    window_start = prediction_time - GAP

    # Windowed events (time-bounded; excludes NaT rows automatically)
    w = events[
        (events["time"] >= window_start) & (events["time"] <= prediction_time)
    ].copy()

    feats: dict = {}

    # ------------------------------------------------------------------ age
    birth_rows = events[events["code"] == cfg["birth_code"]]
    if len(birth_rows) > 0 and pd.notna(birth_rows["time"].iloc[0]):
        birth_ts = birth_rows["time"].iloc[0]
        feats["age"] = (prediction_time - birth_ts).days / 365.25
    else:
        feats["age"] = np.nan

    # ------------------------------------------------------------------ sex
    sex_rows = events[events["code"].str.startswith(cfg["sex_code_prefix"] + "//", na=False)]
    if len(sex_rows) > 0:
        feats["sex_male"] = 1.0 if sex_rows["code"].iloc[0].endswith("//M") else 0.0
    else:
        feats["sex_male"] = np.nan

    # ------------------------------------------------- numeric concepts
    for concept, spec in cfg["numeric_concepts"].items():
        prefixes = spec["codes"]
        temp_f_prefixes = spec.get("temp_f_codes", [])

        mask = code_matches(w["code"], prefixes)
        subset = w[mask].copy()

        # °F → °C conversion for temperature
        if temp_f_prefixes:
            f_mask = code_matches(subset["code"], temp_f_prefixes)
            subset["numeric_value"] = _f2c(subset["numeric_value"], f_mask)

        vals = subset["numeric_value"].dropna()

        if len(vals) > 0:
            feats[f"{concept}_first"]   = vals.iloc[0]
            feats[f"{concept}_last"]    = vals.iloc[-1]
            feats[f"{concept}_min"]     = float(vals.min())
            feats[f"{concept}_max"]     = float(vals.max())
            feats[f"{concept}_mean"]    = float(vals.mean())
            feats[f"{concept}_count"]   = float(len(vals))
            feats[f"{concept}_missing"] = 0.0
        else:
            for s in ["first", "last", "min", "max", "mean"]:
                feats[f"{concept}_{s}"] = np.nan
            feats[f"{concept}_count"]   = 0.0
            feats[f"{concept}_missing"] = 1.0

    # ------------------------------------------------- event family counts
    for family in cfg["event_families"]:
        mask = (w["code"] == family) | w["code"].str.startswith(family + "//", na=False)
        feats[f"n_{family.lower().replace('_', '')}"] = float(mask.sum())

    # ------------------------------------------------- fluid balance
    inp_pfx = cfg.get("input_prefix")
    out_pfx = cfg.get("output_prefix")

    if inp_pfx:
        mask = (w["code"] == inp_pfx) | w["code"].str.startswith(inp_pfx + "//", na=False)
        feats["total_input"]    = float(w.loc[mask, "numeric_value"].fillna(0).sum())
        feats["n_input_events"] = float(mask.sum())

    if out_pfx:
        mask = (w["code"] == out_pfx) | w["code"].str.startswith(out_pfx + "//", na=False)
        feats["total_output"]    = float(w.loc[mask, "numeric_value"].fillna(0).sum())
        feats["n_output_events"] = float(mask.sum())

    # total events in window
    feats["n_events_total"] = float(len(w))

    return feats


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------

COLS_NEEDED = ["subject_id", "time", "code", "numeric_value"]


def process_shard(
    shard_path: Path,
    label_lookup: dict,   # {subject_id: [(pred_time, boolean_value, split), ...]}
    cfg: dict,
) -> list[dict]:
    """Load one MEDS parquet shard and extract features for subjects with labels."""
    try:
        df = pq.read_table(
            shard_path,
            columns=[c for c in COLS_NEEDED if c in pq.read_schema(shard_path).names],
        ).to_pandas()
    except Exception as e:
        log.warning(f"Skipping {shard_path}: {e}")
        return []

    # Ensure numeric_value column exists
    if "numeric_value" not in df.columns:
        df["numeric_value"] = np.nan

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    subjects_in_shard = set(df["subject_id"].unique())
    subjects_with_labels = subjects_in_shard & set(label_lookup.keys())

    if not subjects_with_labels:
        return []

    rows = []
    for sid in subjects_with_labels:
        subj_events = df[df["subject_id"] == sid]
        for pred_time, label, split in label_lookup[sid]:
            feats = compute_features(subj_events, pred_time, cfg)
            feats["subject_id"]      = sid
            feats["prediction_time"] = pred_time
            feats["label"]           = int(label)
            feats["split"]           = split
            rows.append(feats)

    return rows


# ---------------------------------------------------------------------------
# Dataset-aware shard enumeration
# ---------------------------------------------------------------------------

def iter_shards(meds_dir: Path, requested_splits: list[str]) -> list[Path]:
    """Return all parquet shards for the requested splits."""
    shards = []
    for split in requested_splits:
        split_dir = meds_dir / "data" / split
        if split_dir.exists():
            shards.extend(sorted(split_dir.glob("*.parquet")))
        else:
            log.warning(f"Split directory not found: {split_dir}")
    return shards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract first-24h features from MEDS data.")
    parser.add_argument("--meds_dir",    required=True,
                        help="Root MEDS directory (contains data/ and metadata/).")
    parser.add_argument("--labels_path", required=True,
                        help="Labels: directory with split subdirs (K-MIMIC) "
                             "or single .parquet file (MIMIC).")
    parser.add_argument("--splits_path", default=None,
                        help="subject_splits.parquet — required for MIMIC.")
    parser.add_argument("--dataset",    required=True, choices=["kmimic", "mimic"],
                        help="Which concept mapping to use.")
    parser.add_argument("--concepts",   default="experiments/concepts.yaml",
                        help="Path to concepts.yaml.")
    parser.add_argument("--splits",     nargs="+",
                        default=["train", "tuning", "held_out"],
                        help="Splits to process (default: all three).")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where features_{split}.parquet are written.")
    args = parser.parse_args()

    cfg = load_concepts(args.concepts, args.dataset)
    meds_dir = Path(args.meds_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load labels -------------------------------------------------------
    log.info("Loading labels...")
    if args.dataset == "kmimic":
        labels = load_labels_kmimic(args.labels_path)
    else:
        if args.splits_path is None:
            parser.error("--splits_path is required for MIMIC.")
        labels = load_labels_mimic(args.labels_path, args.splits_path)

    # Filter to requested splits
    labels = labels[labels["split"].isin(args.splits)].copy()
    log.info(f"  {len(labels):,} label rows across splits: "
             f"{labels['split'].value_counts().to_dict()}")

    # Build lookup: subject_id → [(pred_time, label, split), ...]
    label_lookup: dict = {}
    for row in labels.itertuples(index=False):
        label_lookup.setdefault(row.subject_id, []).append(
            (row.prediction_time, row.boolean_value, row.split)
        )

    # ---- iterate shards ----------------------------------------------------
    shards = iter_shards(meds_dir, args.splits)
    log.info(f"Processing {len(shards)} shards...")

    all_rows = []
    for i, shard in enumerate(shards):
        rows = process_shard(shard, label_lookup, cfg)
        all_rows.extend(rows)
        if (i + 1) % 50 == 0 or (i + 1) == len(shards):
            log.info(f"  {i+1}/{len(shards)} shards done — {len(all_rows):,} rows so far")

    if not all_rows:
        log.error("No features extracted. Check paths and dataset name.")
        return

    # ---- assemble & save ---------------------------------------------------
    features_df = pd.DataFrame(all_rows)
    log.info(f"Total: {len(features_df):,} rows, {features_df.shape[1]} columns")

    for split in args.splits:
        subset = features_df[features_df["split"] == split].reset_index(drop=True)
        if subset.empty:
            continue
        out_path = output_dir / f"features_{split}.parquet"
        subset.to_parquet(out_path, index=False)
        pos = subset["label"].sum()
        log.info(f"  {split}: {len(subset):,} rows, "
                 f"{int(pos)} positives ({pos/len(subset)*100:.1f}%)  → {out_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
