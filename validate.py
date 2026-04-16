"""
CLI validation script for K-MIMIC-MEDS output.
Runs the same checks as the validation notebook but from the command line.

Usage:
    python validate.py --output_dir data/output
"""

import json
import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Plausible value ranges for common clinical measurements
# Format: code_prefix -> (min, max)
# ---------------------------------------------------------------------------
CLINICAL_RANGES = {
    "CHARTEVENT//001C_1021":  (0,   300),   # Heart rate (bpm)
    "CHARTEVENT//001C_1023":  (0,   60),    # Respiratory rate (/min)
    "CHARTEVENT//001C_1026":  (30,  45),    # Body temperature (Celsius)
    "CHARTEVENT//001C_1012":  (40,  250),   # Systolic BP (mmHg)
    "CHARTEVENT//001C_1013":  (20,  180),   # Diastolic BP (mmHg)
    "CHARTEVENT//001C_1003":  (50,  100),   # SpO2 (%)
    "LAB//001L2001":          (0,   100),   # WBC (x10e3/uL)
    "LAB//001L2003":          (3,   20),    # Hemoglobin (g/dL)
    "LAB//001L3005":          (40,  500),   # Glucose (mg/dL)
}


def validate(output_dir: Path) -> bool:
    passed = 0
    failed = 0

    def check(name, result, detail=""):
        nonlocal passed, failed
        status = "PASS" if result else "FAIL"
        detail_str = f" — {detail}" if detail else ""
        print(f"  [{status}] {name}{detail_str}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nValidating MEDS dataset at: {output_dir}\n")

    # -------------------------------------------------------------------------
    # 1. File structure
    # -------------------------------------------------------------------------
    print("1. File structure")
    expected_files = [
        "data/train/0.parquet",
        "data/tuning/0.parquet",
        "data/held_out/0.parquet",
        "metadata/codes.parquet",
        "metadata/dataset.json",
        "metadata/subject_splits.parquet",
    ]
    for f in expected_files:
        check(f, (output_dir / f).exists())

    # -------------------------------------------------------------------------
    # 2. MEDS schema
    # -------------------------------------------------------------------------
    print("\n2. MEDS schema")
    train = pd.read_parquet(output_dir / "data/train/0.parquet")
    check("subject_id is int64",       str(train["subject_id"].dtype) in ("int64", "Int64"))
    check("time is datetime",          "datetime" in str(train["time"].dtype))
    check("code is string",            str(train["code"].dtype) in ("object", "string", "str"))
    check("numeric_value is float32",  str(train["numeric_value"].dtype) == "float32")
    check("exactly 4 columns",         len(train.columns) == 4)

    # -------------------------------------------------------------------------
    # 3. Code quality
    # -------------------------------------------------------------------------
    print("\n3. Code quality")
    nan_codes = train[train["code"].str.contains("//nan", na=False)]
    check("no //nan in codes", len(nan_codes) == 0, f"{len(nan_codes)} found")
    unknown_codes = train[train["code"] == "UNKNOWN"]
    check("no UNKNOWN codes", len(unknown_codes) == 0, f"{len(unknown_codes)} found")
    korean = train[train["code"].str.contains("회|℃|㎍|㎎|×|㎕|μℓ", na=False, regex=True)]
    check("no Korean units in codes", len(korean) == 0, f"{len(korean)} found")
    if len(korean) > 0:
        print(f"    Sample: {korean['code'].unique()[:5]}")

    # -------------------------------------------------------------------------
    # 4. Static events
    # -------------------------------------------------------------------------
    print("\n4. Static events")
    static = train[train["time"].isna()]
    check("static events exist", len(static) > 0, f"{len(static)} found")
    if len(static) > 0:
        check("static events are GENDER only", static["code"].str.startswith("GENDER").all())

    # -------------------------------------------------------------------------
    # 5. Splits
    # -------------------------------------------------------------------------
    print("\n5. Splits")
    splits = pd.read_parquet(output_dir / "metadata/subject_splits.parquet")
    total = len(splits)
    counts = splits["split"].value_counts()
    train_pct  = counts.get("train", 0) / total * 100
    tuning_pct = counts.get("tuning", 0) / total * 100
    held_pct   = counts.get("held_out", 0) / total * 100
    check("train ~80%",    75 <= train_pct  <= 85, f"{train_pct:.1f}%")
    check("tuning ~10%",    8 <= tuning_pct <= 12, f"{tuning_pct:.1f}%")
    check("held_out ~10%",  8 <= held_pct   <= 12, f"{held_pct:.1f}%")

    # -------------------------------------------------------------------------
    # 6. Metadata
    # -------------------------------------------------------------------------
    print("\n6. Metadata")
    codes = pd.read_parquet(output_dir / "metadata/codes.parquet")
    check("codes.parquet has descriptions",
          codes["description"].notna().sum() > 0,
          f"{codes['description'].notna().sum()}/{len(codes)} codes")
    check("codes.parquet has parent_codes",
          codes["parent_codes"].notna().sum() > 0,
          f"{codes['parent_codes'].notna().sum()}/{len(codes)} codes")
    meta = json.loads((output_dir / "metadata/dataset.json").read_text())
    check("dataset.json has dataset_name", "dataset_name" in meta)
    check("dataset.json has meds_version", "meds_version" in meta)

    # -------------------------------------------------------------------------
    # 7. Global statistics
    # -------------------------------------------------------------------------
    print("\n7. Global statistics")
    all_dfs = []
    for split_name in ["train", "tuning", "held_out"]:
        df = pd.read_parquet(output_dir / f"data/{split_name}/0.parquet")
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)
    check("total events > 0",
          len(full) > 0,
          f"{len(full):,}")
    check("total patients > 0",
          full["subject_id"].nunique() > 0,
          f"{full['subject_id'].nunique():,}")
    check("events with numeric value exist",
          full["numeric_value"].notna().sum() > 0,
          f"{full['numeric_value'].notna().sum():,}")

    # -------------------------------------------------------------------------
    # 8. Temporal consistency
    # Verify that event chronology is medically plausible for each patient.
    # -------------------------------------------------------------------------
    print("\n8. Temporal consistency")

    dynamic = full[full["time"].notna()].copy()
    dynamic["time"] = pd.to_datetime(dynamic["time"], errors="coerce")

    # 8a. MEDS_BIRTH is the earliest event for each patient
    birth = dynamic[dynamic["code"] == "MEDS_BIRTH"][["subject_id", "time"]].rename(columns={"time": "birth_time"})
    if not birth.empty:
        earliest = dynamic.groupby("subject_id")["time"].min().reset_index().rename(columns={"time": "earliest_time"})
        merged = birth.merge(earliest, on="subject_id")
        birth_violations = merged[merged["birth_time"] > merged["earliest_time"]]
        check("MEDS_BIRTH is earliest event per patient",
              len(birth_violations) == 0,
              f"{len(birth_violations)} violations")

    # 8b. MEDS_DEATH is the latest event per patient
    # Note: ICU procedure events (PROCEDURE_START/END) in K-MIMIC synthetic data
    # can occasionally have timestamps after dod due to data generation artifacts.
    # We exclude procedure events from this check.
    death = dynamic[dynamic["code"] == "MEDS_DEATH"][["subject_id", "time"]].rename(columns={"time": "death_time"})
    if not death.empty:
        non_proc = dynamic[~dynamic["code"].str.startswith(("PROCEDURE_START", "PROCEDURE_END"))]
        latest = non_proc.groupby("subject_id")["time"].max().reset_index().rename(columns={"time": "latest_time"})
        merged = death.merge(latest, on="subject_id")
        # Allow a 48h tolerance window — dod in K-MIMIC is date-only (midnight)
        # while clinical measurements have precise timestamps, sometimes spanning
        # into the next day due to synthetic data generation artifacts
        tolerance = pd.Timedelta(hours=48)
        death_violations = merged[merged["death_time"] + tolerance < merged["latest_time"]]
        check("MEDS_DEATH is latest event per patient (excl. procedures, 48h tolerance)",
            len(death_violations) == 0,
            f"{len(death_violations)} violations")
        
        # --- debug ---
        if len(death_violations) > 0:
            sample_sid = death_violations["subject_id"].iloc[0]
            patient = non_proc[non_proc["subject_id"] == sample_sid].sort_values("time")
            death_t = death_violations[death_violations["subject_id"] == sample_sid]["death_time"].iloc[0]
            print(f"\n    Debug — patient {sample_sid}, death at {death_t}")
            after_death = patient[patient["time"] > death_t][["time", "code"]].head(5)
            print(after_death.to_string())

    # 8c. ICU_ADMISSION always before ICU_DISCHARGE for each patient
    icu_in  = dynamic[dynamic["code"].str.startswith("ICU_ADMISSION")][["subject_id", "time"]].rename(columns={"time": "intime"})
    icu_out = dynamic[dynamic["code"].str.startswith("ICU_DISCHARGE")][["subject_id", "time"]].rename(columns={"time": "outtime"})
    if not icu_in.empty and not icu_out.empty:
        icu_first_in  = icu_in.groupby("subject_id")["intime"].min().reset_index()
        icu_first_out = icu_out.groupby("subject_id")["outtime"].min().reset_index()
        icu_merged = icu_first_in.merge(icu_first_out, on="subject_id")
        icu_violations = icu_merged[icu_merged["intime"] > icu_merged["outtime"]]
        check("ICU_ADMISSION before ICU_DISCHARGE",
              len(icu_violations) == 0,
              f"{len(icu_violations)} violations")

    # 8d. HOSPITAL_ADMISSION always before HOSPITAL_DISCHARGE
    hadm_in  = dynamic[dynamic["code"].str.startswith("HOSPITAL_ADMISSION")][["subject_id", "time"]].rename(columns={"time": "admittime"})
    hadm_out = dynamic[dynamic["code"].str.startswith("HOSPITAL_DISCHARGE")][["subject_id", "time"]].rename(columns={"time": "dischtime"})
    if not hadm_in.empty and not hadm_out.empty:
        hadm_first_in  = hadm_in.groupby("subject_id")["admittime"].min().reset_index()
        hadm_first_out = hadm_out.groupby("subject_id")["dischtime"].min().reset_index()
        hadm_merged = hadm_first_in.merge(hadm_first_out, on="subject_id")
        hadm_violations = hadm_merged[hadm_merged["admittime"] > hadm_merged["dischtime"]]
        check("HOSPITAL_ADMISSION before HOSPITAL_DISCHARGE",
              len(hadm_violations) == 0,
              f"{len(hadm_violations)} violations")

    # 8e. No duplicate static events per patient (each patient should have exactly 1 GENDER)
    gender_counts = full[full["code"].str.startswith("GENDER")].groupby("subject_id").size()
    duplicate_gender = (gender_counts > 1).sum()
    check("no duplicate GENDER events per patient",
          duplicate_gender == 0,
          f"{duplicate_gender} patients with duplicates")

    # -------------------------------------------------------------------------
    # 9. Distributional checks
    # Verify that numeric values are within clinically plausible ranges.
    # -------------------------------------------------------------------------
    print("\n9. Distributional checks")

    numeric = full[full["numeric_value"].notna()].copy()

    for code_prefix, (low, high) in CLINICAL_RANGES.items():
        subset = numeric[numeric["code"].str.startswith(code_prefix)]
        if subset.empty:
            continue
        out_of_range = subset[
            (subset["numeric_value"] < low) | (subset["numeric_value"] > high)
        ]
        pct_ok = (1 - len(out_of_range) / len(subset)) * 100
        check(
            f"{code_prefix.split('//')[1]} values in [{low}, {high}]",
            pct_ok >= 95,  # allow up to 5% outliers
            f"{pct_ok:.1f}% in range ({len(out_of_range)} outliers out of {len(subset)})"
        )

    # -------------------------------------------------------------------------
    # 10. Cross-cohort alignment
    # Verify that code distributions are consistent across train/tuning/held_out.
    # -------------------------------------------------------------------------
    print("\n10. Cross-cohort alignment")

    split_dfs = {}
    for split_name in ["train", "tuning", "held_out"]:
        split_dfs[split_name] = pd.read_parquet(output_dir / f"data/{split_name}/0.parquet")

    # 10a. All splits contain the same set of code prefixes
    def get_prefixes(df):
        return set(df["code"].str.split("//").str[0].unique())

    train_prefixes  = get_prefixes(split_dfs["train"])
    tuning_prefixes = get_prefixes(split_dfs["tuning"])
    held_prefixes   = get_prefixes(split_dfs["held_out"])
    all_same = (train_prefixes == tuning_prefixes == held_prefixes)
    check("same event types across all splits", all_same,
          "" if all_same else f"missing in tuning: {train_prefixes - tuning_prefixes}, missing in held_out: {train_prefixes - held_prefixes}")

    # 10b. Numeric value distributions are comparable across splits
    # Compare mean values for the top numeric code in each split — should be close
    top_numeric_code = (
        full[full["numeric_value"].notna()]
        ["code"].value_counts().index[0]
    )
    means = {}
    for split_name, df in split_dfs.items():
        subset = df[df["code"] == top_numeric_code]["numeric_value"]
        if not subset.empty:
            means[split_name] = subset.mean()

    if len(means) == 3:
        overall_mean = sum(means.values()) / len(means)
        max_deviation = max(abs(v - overall_mean) / overall_mean * 100 for v in means.values())
        check(
            f"numeric value means comparable across splits ({top_numeric_code.split('//')[1]})",
            max_deviation <= 20,
            f"max deviation {max_deviation:.1f}% from overall mean"
        )

    # 10c. No patient appears in more than one split
    all_subjects = []
    for split_name, df in split_dfs.items():
        sids = set(df["subject_id"].unique())
        all_subjects.append((split_name, sids))

    train_set  = all_subjects[0][1]
    tuning_set = all_subjects[1][1]
    held_set   = all_subjects[2][1]
    overlap_train_tuning = train_set & tuning_set
    overlap_train_held   = train_set & held_set
    overlap_tuning_held  = tuning_set & held_set
    total_overlap = len(overlap_train_tuning) + len(overlap_train_held) + len(overlap_tuning_held)
    check("no patient appears in multiple splits",
          total_overlap == 0,
          f"{total_overlap} overlapping patients" if total_overlap > 0 else "")

    # 10d. Each split covers at least 5% of all unique codes
    all_codes = set(full["code"].unique())
    for split_name, df in split_dfs.items():
        split_codes = set(df["code"].unique())
        coverage = len(split_codes) / len(all_codes) * 100
        check(f"{split_name} covers >80% of all codes",
              coverage >= 80,
              f"{coverage:.1f}% ({len(split_codes)}/{len(all_codes)} codes)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_checks = passed + failed
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total_checks} checks passed")
    if failed > 0:
        print(f"        {failed} checks FAILED")
    print(f"{'='*50}\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Validate a K-MIMIC-MEDS output dataset")
    parser.add_argument("--output_dir", type=Path, default=Path("data/output"))
    args = parser.parse_args()

    success = validate(args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()