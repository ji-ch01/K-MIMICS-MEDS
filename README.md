# K-MIMIC-MEDS

ETL pipeline to convert the **Synthetic K-MIMIC (SYN-ICU)** Korean ICU dataset into the **MEDS** (Medical Event Data Standard) format, with a transportable in-hospital mortality benchmark.

> **Paper:** *Synthetic K-MIMIC in MEDS: Validation and a Transportable Mortality Benchmark* — submitted to the SD4H Workshop @ ICML 2026.

This project was developed as part of an internship at **[VitalLab](https://sites.google.com/vitaldb.net/vitallab-snucmsnuh/home)** — Department of Anesthesiology and Pain Medicine, Seoul National University College of Medicine / Seoul National University Hospital (SNUCM/SNUH) — under the supervision of Professor Hyung-Chul Lee and Professor Hyeonhoon Lee.

The intern is a student at **Efrei Paris** (M2 Bioinformatics, Data Engineering & AI).

This project is part of a research initiative to extend the MEDS ecosystem to non-English clinical datasets, starting with the Korean ICU synthetic dataset published by KHDP.

---

## Overview

[MEDS](https://medical-event-data-standard.github.io/) is a minimal, interoperable data standard for longitudinal medical event data, designed for reproducible machine learning research in healthcare.

This repository provides a **standalone ETL pipeline** (no MEDS-Extract CLI required) that:

1. Cleans and transforms raw K-MIMIC `.xlsx` tables into intermediate Parquet files (`pre_meds.py`)
2. Converts those Parquet files into a fully MEDS-compliant dataset (`meds_convert.py`)
3. Validates the output with a Jupyter notebook (`validation.ipynb`) and a CLI script (`validate.py`)
4. Extracts binary task labels in MEDS-DEV format (`extract_labels.py`)

The ETL code is packaged as a Python package (`kmimic_meds`) intended for publication on PyPI.

> **Note:** MEDS-Extract CLI was evaluated but not used due to Windows/Hydra compatibility issues. The standalone approach produces identical output and works cross-platform.

---

## Data Sources

| Source                                                                              | Description                                           |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [Synthetic K-MIMIC (SYN-ICU)](https://khdp.net/database/data-search-detail/SYN-ICU) | Synthetic Korean ICU dataset published by KHDP        |
| [MIMIC-IV-MEDS](https://physionet.org/content/mimic-iv-demo-meds/0.0.1/)            | Reference MEDS conversion of MIMIC-IV (used as model) |

> Raw K-MIMIC data is **not included** in this repository. Download it from the KHDP portal and place the `.xlsx` files under `data/raw/`.

---

## Project Structure

```
K-MIMIC-MEDS/
├── .github/
│   └── workflows/
│       └── tests.yml                     # GitHub Actions — runs pytest on push
├── data/
│   ├── raw/                              # Raw K-MIMIC .xlsx files (not versioned)
│   ├── intermediate/                     # Pre-MEDS Parquet files (not versioned)
│   ├── output/                           # Final MEDS-compliant dataset (not versioned)
│   ├── labels/                           # Extracted task labels in MEDS-DEV format (not versioned)
│   └── kmimic_triplet_tensors/           # meds-torch NRT tensors (not versioned)
├── experiments/
│   ├── lane_a/
│   │   ├── preprocess_kmimic.py          # Standalone preprocessing pipeline → NRT tensors
│   │   ├── configs/
│   │   │   ├── kmimic_train.yaml         # meds-torch training config (K-MIMIC)
│   │   │   └── mimic_train.yaml          # meds-torch training config (MIMIC-IV)
│   │   └── run_lane_a.sh                 # End-to-end Lane A script
│   ├── lane_b/
│   │   ├── feature_extract.py            # 24h feature extraction (77 K-MIMIC / 76 MIMIC-IV features)
│   │   ├── train_xgb.py                  # XGBoost training — within and cross-cohort
│   │   ├── features/
│   │   │   ├── kmimic/                   # K-MIMIC feature matrices (versioned)
│   │   │   └── mimic/                    # MIMIC-IV feature matrices (not versioned — DUA)
│   │   ├── results/
│   │   │   ├── metrics.json              # AUROC / AUPRC / Brier point estimates
│   │   │   ├── bootstrap_ci.json         # 95% bootstrap CIs (n=2000)
│   │   │   ├── predictions_kmimic_within.parquet
│   │   │   └── predictions_mimic_to_kmimic.parquet
│   │   └── run_lane_b.sh                 # End-to-end Lane B script
│   └── concepts.yaml                     # Shared feature concept definitions
├── src/
│   └── kmimic_meds/
│       ├── etl/
│       │   ├── pre_meds.py               # Step 1 — clean raw .xlsx → intermediate .parquet
│       │   └── meds_convert.py           # Step 2 — intermediate .parquet → MEDS dataset
│       └── utils/
│           └── io.py
├── tests/
│   └── test_meds_convert.py              # 71 unit tests
├── bootstrap.py                          # Bootstrap CI computation for benchmark metrics
├── validation.ipynb                      # Validation notebook (24/24 checks + 3 extended sections)
├── validate.py                           # CLI validation script (46/46 checks)
├── extract_labels.py                     # Label extraction — in-hospital mortality 24h (MEDS-DEV format)
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ji-ch01/K-MIMIC-MEDS.git
cd K-MIMIC-MEDS
pip install -e .
pip install openpyxl
```

---

## Usage

### Step 1 — Place raw data

Download the Synthetic K-MIMIC dataset from [KHDP](https://khdp.net/database/data-search-detail/SYN-ICU) and place all `.xlsx` files under `data/raw/`.

Expected files:

```
data/raw/
├── syn_admissions.xlsx
├── syn_chartevents.xlsx
├── syn_d_items.xlsx
├── syn_d_labitems.xlsx
├── syn_diagnoses_icd.xlsx
├── syn_emar.xlsx
├── syn_emar_detail.xlsx
├── syn_icustays.xlsx
├── syn_inputevents.xlsx
├── syn_labevents.xlsx
├── syn_outputevents.xlsx
├── syn_patients.xlsx
├── syn_procedureevents.xlsx
├── syn_procedures_icd.xlsx
└── syn_transfers.xlsx
```

### Step 2 — Run Pre-MEDS

Transforms raw `.xlsx` files into cleaned `.parquet` files.

```bash
python src/kmimic_meds/etl/pre_meds.py \
    --input_dir data/raw \
    --output_dir data/intermediate
```

What it does:

- Reads all 15 `.xlsx` source tables
- Converts UUID string IDs to stable `int64` via SHA-256 hashing (collision-free verified)
- Parses and normalizes timestamps (including mixed date/datetime formats)
- Computes `year_of_birth = anchor_year - anchor_age`
- Resolves date-only columns (adds `23:59:59` to `chartdate` in `procedures_icd`)
- Renames `icustay_id` → `stay_id` in `inputevents`/`outputevents`
- Casts free-text `value` columns to string

### Step 3 — Run MEDS Conversion

Converts intermediate Parquet files into the final MEDS-compliant dataset.

```bash
python src/kmimic_meds/etl/meds_convert.py \
    --intermediate_dir data/intermediate \
    --output_dir data/output
```

What it does:

- Extracts MEDS events from each source table (vectorized, ~7s total)
- Normalizes Korean and non-standard units to international equivalents
- Links diagnoses to admission timestamps via `hadm_id` join
- Assigns patients to `train` (80%), `tuning` (10%), `held_out` (10%) splits with fixed seed
- Writes Parquet files with the strict MEDS PyArrow schema
- Enriches `codes.parquet` with descriptions and EDI parent codes

### Step 4 — Validate

**Option A — Jupyter notebook** (visual, for presentations):

```bash
pip install notebook
jupyter notebook validation.ipynb
```

Expected result: **24/24 checks passed** (core summary), plus three extended validation sections:
- §20 Source-to-MEDS coverage table (15 source tables audited)
- §21 Code overlap analysis (204 exact codes, 20 families, 32 EDI parent codes; set `MIMIC_MEDS_DIR` to run cross-cohort comparison)
- §22 Tensorisation timestamp smoke test (datetime64[us] round-trip verified, 74% of events have year > 2262)

**Option B — CLI script** (fast, for CI):

```bash
python validate.py --output_dir data/output
```

Expected result: **46/46 checks passed**.

**Option C — Unit tests** (for code correctness):

```bash
pytest tests/test_meds_convert.py -v
```

Expected result: **71/71 tests passed**.

### Step 5 — Extract task labels

Extracts binary mortality labels in [MEDS-DEV](https://github.com/Medical-Event-Data-Standard/meds_evaluation) compatible format.

```bash
python extract_labels.py
```

Two candidate tasks are evaluated; **in-hospital mortality** is selected as the primary task:

| Task | Cohort | Positives | Prevalence | Selected |
|---|---|---|---|---|
| `icu_mortality_24h` | 522 | 2 | 0.4% | No — too sparse |
| `inhospital_mortality_24h` | 957 | 81 | 8.5% | **Yes** |

**Definition:** prediction at hospital admission + 24h; label = `MEDS_DEATH` before `HOSPITAL_DISCHARGE`. Patients discharged or deceased within 24h of admission are excluded.

Output:

```
data/labels/
└── inhospital_mortality_24h/
    ├── train/0.parquet     ← 767 patients, 67 positives (8.7%)
    ├── tuning/0.parquet    ← 88 patients,   6 positives (6.8%)
    └── held_out/0.parquet  ← 102 patients,  8 positives (7.8%)
```

Each file contains: `subject_id (int64)` | `prediction_time (timestamp[us])` | `boolean_value (bool)`.

---

## Mortality Prediction Benchmark

The `experiments/` directory contains two benchmark lanes for in-hospital mortality prediction (24h observation window, 81 positives / 957 patients, 8.5% prevalence).

### Lane A — MEDS-native (meds-torch)

Trains a Transformer (token_dim=64, 2 layers, 4 heads, 214K parameters) using [meds-torch 0.0.8](https://github.com/Oufattole/meds-torch).

```bash
# Preprocess K-MIMIC into NRT tensor format
python experiments/lane_a/preprocess_kmimic.py

# Train
meds-torch-train \
  --config-dir experiments/lane_a/configs \
  --config-name kmimic_train \
  'hydra.searchpath=[pkg://meds_torch.configs]'
```

**Result:** Both configurations (no reweighting and pos_weight=10.8) fail to learn a useful ranking signal (AUROC 0.266 / 0.234), consistent with known deep learning limitations on small clinical cohorts (~65 positive training examples). This constitutes an informative lower bound motivating feature-based methods.

### Lane B — Cross-cohort transfer (XGBoost)

Builds a shared 24h feature representation (73 features shared across K-MIMIC and MIMIC-IV) and trains XGBoost within-dataset and cross-cohort.

```bash
python experiments/lane_b/feature_extract.py   # extract features
python experiments/lane_b/train_xgb.py         # train and evaluate
python bootstrap.py                            # compute 95% CIs
```

### Results

| Model | Train | AUROC | 95% CI | AUPRC | 95% CI |
|---|---|---|---|---|---|
| XGBoost | K-MIMIC | **0.810** | [0.631–0.960] | **0.505** | [0.116–0.826] |
| XGBoost | MIMIC-IV→K-MIMIC | 0.674 | [0.418–0.900] | 0.287 | [0.066–0.617] |
| meds-torch | K-MIMIC | 0.266 | — | 0.062 | — |
| meds-torch+pw | K-MIMIC | 0.234 | — | — | — |

Bootstrap CIs (n=2000) on held-out set (102 patients, 8 positives). The 0.136 AUROC gap between within-dataset and transfer quantifies the vocabulary mismatch cost between MIMIC-IV item IDs and K-MIMIC EDI codes.

---

## MEDS Output Format

```
data/output/
├── data/
│   ├── train/0.parquet       ← 80% of patients
│   ├── tuning/0.parquet      ← 10% of patients
│   └── held_out/0.parquet    ← 10% of patients
└── metadata/
    ├── codes.parquet         ← unique codes with descriptions and EDI parent codes
    ├── dataset.json          ← dataset metadata
    └── subject_splits.parquet
```

Each row in the data files follows the MEDS schema:

| Column          | Type            | Description                                                  |
| --------------- | --------------- | ------------------------------------------------------------ |
| `subject_id`    | `int64`         | Unique patient identifier                                    |
| `time`          | `timestamp[us]` | Timestamp of the event (`null` for static events)            |
| `code`          | `string`        | Event code (e.g. `CHARTEVENT//001C_102//mmHg`, `MEDS_BIRTH`) |
| `numeric_value` | `float32`       | Numeric value if applicable, otherwise `null`                |

### Event types and codes

| Prefix               | Source table            | Example code                               |
| -------------------- | ----------------------- | ------------------------------------------ |
| `MEDS_BIRTH`         | `syn_patients`          | `MEDS_BIRTH`                               |
| `MEDS_DEATH`         | `syn_patients`          | `MEDS_DEATH`                               |
| `GENDER`             | `syn_patients` (static) | `GENDER//M`                                |
| `HOSPITAL_ADMISSION` | `syn_admissions`        | `HOSPITAL_ADMISSION//Emergency department` |
| `HOSPITAL_DISCHARGE` | `syn_admissions`        | `HOSPITAL_DISCHARGE//Home`                 |
| `ICU_ADMISSION`      | `syn_icustays`          | `ICU_ADMISSION//RICU`                      |
| `ICU_DISCHARGE`      | `syn_icustays`          | `ICU_DISCHARGE//RICU`                      |
| `CHARTEVENT`         | `syn_chartevents`       | `CHARTEVENT//001C_1021_25105///min`        |
| `LAB`                | `syn_labevents`         | `LAB//001L3005//mg/dL`                     |
| `DIAGNOSIS`          | `syn_diagnoses_icd`     | `DIAGNOSIS//KCD8//I251`                    |
| `PROCEDURE_ICD`      | `syn_procedures_icd`    | `PROCEDURE_ICD//KCD8//54.11`               |
| `PROCEDURE_START`    | `syn_procedureevents`   | `PROCEDURE_START//001P_OPFG130303`         |
| `PROCEDURE_END`      | `syn_procedureevents`   | `PROCEDURE_END//001P_OPFG130303`           |
| `INPUT_START`        | `syn_inputevents`       | `INPUT_START//001I_1315//cc`               |
| `OUTPUT`             | `syn_outputevents`      | `OUTPUT//001O_148//cc`                     |
| `MEDICATION`         | `syn_emar`              | `MEDICATION//12005122`                     |

### Dataset statistics (SYN-ICU)

| Metric                      | Value       |
| --------------------------- | ----------- |
| Total events                | 1,381,580   |
| Total patients              | 1,328       |
| Static events (time = null) | 1,328       |
| Dynamic events              | 1,380,252   |
| Events with numeric value   | 605,941     |
| Unique codes                | 204         |
| Codes with description      | 113         |
| Codes with EDI parent codes | 32          |
| Train patients              | 1,062 (80%) |
| Tuning patients             | 132 (10%)   |
| Held-out patients           | 134 (10%)   |
| Pipeline runtime            | ~7s         |

---

## Key Design Decisions

**Alignment with MEDS KDD 2025 tutorial:** The pipeline follows the conceptual framework from the official [MEDS "Converting to MEDS" tutorial (KDD 2025)](https://medical-event-data-standard.github.io/docs/tutorials/kdd2025/converting_to_MEDS) — two-step Pre-MEDS + conversion approach, `//` code separator, date-only timestamp resolution, and `deathtime` priority over `dod` are all directly aligned with the tutorial's recommendations.

**UUID → int64 conversion:** K-MIMIC uses UUID strings for all identifiers. MEDS requires `int64`. We use SHA-256 hashing to produce stable, positive int64 values. Uniqueness is verified after conversion with an assertion — for 1,328 patients the birthday bound is ~2³¹, making collisions negligible. The stateless approach requires no central mapping table and works identically across all 15 source tables.

**Standalone pipeline:** MEDS-Extract CLI was evaluated but bypassed due to Windows/Hydra compatibility issues. The standalone `meds_convert.py` produces identical output with no external CLI dependency.

**Vectorized extractors:** All event extraction functions use pandas vectorized operations instead of `iterrows()`, reducing runtime from ~40s to ~7s (10-50x faster).

**Diagnosis timestamps:** Diagnoses ICD are joined with the admissions table on `hadm_id` to recover their admission timestamp, making them dynamic events rather than static ones. All 3,091 diagnoses have real timestamps.

**Date-only timestamps:** `procedures_icd.chartdate` is a date without time. We add `23:59:59` to avoid temporal leakage in prediction tasks — placing the procedure at the end of the recorded day.

**Korean unit normalization:** K-MIMIC uses Korean and non-standard units (`회/min`, `℃`, `×10³/㎕`, etc.). All units are mapped to international equivalents via a `UNIT_MAP` dictionary. Result: 0 non-standard units remaining.

**Korean medical codes:** K-MIMIC uses the Korean Classification of Disease (KCD8) standard for diagnoses. These codes are preserved as-is and linked to EDI parent codes for lab items.

**Microsecond timestamps:** K-MIMIC uses de-identified future years (2795+) that exceed the pandas nanosecond limit (year 2262). All timestamps use `datetime64[us]` precision instead, extending the range to year 294,000.

---

## Known Limitations

**Synthetic dataset:** SYN-ICU is a synthetically generated dataset. Its distributions do not reflect real Korean ICU patient populations. Models trained on this dataset should not be expected to generalize directly to real clinical settings without further validation on real data.

**`MEDS_DEATH` temporal precision:** The `dod` field in K-MIMIC is stored as a date only (midnight, `00:00:00`). When `admissions.deathtime` is available (88 out of 88 deceased patients), the pipeline uses that field instead, providing hour-level precision. For patients where `deathtime` is absent, `dod` midnight is used as a fallback. The temporal consistency check applies a 48-hour tolerance window and excludes `PROCEDURE_START`/`PROCEDURE_END` events, which are a known artifact of the synthetic data generation process.

**Missing `MEDS_BIRTH` events:** 79 out of 1,328 patients have no `MEDS_BIRTH` event because `anchor_age` is `NaN` in the source `syn_patients` table — making it impossible to compute `year_of_birth`. These patients are fully retained in the dataset with all their other events (79,165 total).

**Label sparsity in smaller splits:** The in-hospital mortality task has 6 positives in the tuning split and 8 in held_out. This is expected for a 1,328-patient synthetic cohort and should be considered when interpreting AUROC/AUPRC estimates on those splits. ICU mortality (2 positives total) was evaluated and rejected as too sparse for reliable benchmarking.

**`chartdate` precision:** Procedure ICD timestamps are resolved to `23:59:59` on the recorded date. This is a convention to avoid temporal leakage — the exact time of the procedure within the day is not available in the source data.

---

## References

- [MEDS Documentation](https://medical-event-data-standard.github.io/)
- [MEDS GitHub](https://github.com/Medical-Event-Data-Standard/meds)
- [MEDS-Transforms](https://github.com/mmcdermott/MEDS_transforms)
- [MIMIC-IV-MEDS reference ETL](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS)
- [KHDP SYN-ICU Dataset](https://khdp.net/database/data-search-detail/SYN-ICU)
- [K-MIMIC Documentation](https://sites.google.com/view/k-mimic/modules)

---

## License

The Synthetic K-MIMIC (SYN-ICU) dataset is provided by the **NSTRI Data Innovation Center** via the KHDP (Korean Health Data Platform) portal. Please refer to the [KHDP data usage terms](https://khdp.net/database/data-search-detail/SYN-ICU) before using this dataset. The pipeline code is independently developed and not affiliated with KHDP or NSTRI.
