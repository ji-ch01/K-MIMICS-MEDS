# K-MIMIC-MEDS

ETL pipeline to convert the **Synthetic K-MIMIC (SYN-ICU)** Korean ICU dataset into the **MEDS** (Medical Event Data Standard) format.

This project is part of a research initiative to extend the MEDS ecosystem to non-English clinical datasets, starting with the Korean ICU synthetic dataset published by KHDP.

---

## Overview

[MEDS](https://medical-event-data-standard.github.io/) is a minimal, interoperable data standard for longitudinal medical event data, designed for reproducible machine learning research in healthcare.

This repository provides a **standalone ETL pipeline** (no MEDS-Extract CLI required) that:

1. Cleans and transforms raw K-MIMIC `.xlsx` tables into intermediate Parquet files (`pre_meds.py`)
2. Converts those Parquet files into a fully MEDS-compliant dataset (`meds_convert.py`)
3. Validates the output with a Jupyter notebook (`validation.ipynb`)

The ETL code is packaged as a Python package (`kmimic_meds`) intended for publication on PyPI.

> **Note:** MEDS-Extract CLI was evaluated but not used due to Windows compatibility issues with Hydra configuration. The standalone approach produces identical output and works cross-platform.

---

## Data Sources

| Source | Description |
|--------|-------------|
| [Synthetic K-MIMIC (SYN-ICU)](https://khdp.net/database/data-search-detail/SYN-ICU) | Synthetic Korean ICU dataset published by KHDP |
| [MIMIC-IV-MEDS](https://physionet.org/content/mimic-iv-demo-meds/0.0.1/) | Reference MEDS conversion of MIMIC-IV (used as model) |

> Raw K-MIMIC data is **not included** in this repository. Download it from the KHDP portal and place the `.xlsx` files under `data/raw/`.

---

## Project Structure

```
K-MIMIC-MEDS/
├── configs/
│   └── messy.yaml              # MEDS-Extract event mapping (reference, not used in pipeline)
├── data/
│   ├── raw/                    # Raw K-MIMIC .xlsx files (not versioned)
│   ├── intermediate/           # Pre-MEDS Parquet files (not versioned)
│   └── output/                 # Final MEDS-compliant dataset (not versioned)
├── src/
│   └── kmimic_meds/
│       ├── etl/
│       │   ├── pre_meds.py     # Step 1 — clean raw .xlsx → intermediate .parquet
│       │   └── meds_convert.py # Step 2 — intermediate .parquet → MEDS dataset
│       └── utils/
│           └── io.py
├── tests/
│   └── test_pre_meds.py
├── validation.ipynb            # Validation notebook (9/9 checks)
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/K-MIMIC-MEDS.git
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
- Converts UUID string IDs to stable `int64` via SHA-256 hashing
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
- Extracts MEDS events from each source table
- Builds hierarchical codes using `//` as separator (e.g. `CHARTEVENT//001C_102//mmHg`)
- Assigns patients to `train` (80%), `tuning` (10%), `held_out` (10%) splits
- Writes Parquet files with the strict MEDS PyArrow schema
- Enriches `codes.parquet` with descriptions from `syn_d_items` and `syn_d_labitems`

### Step 4 — Validate

Open `validation.ipynb` in Jupyter and run all cells.

```bash
pip install notebook
jupyter notebook validation.ipynb
```

Expected result: **9/9 checks passed**.

---

## MEDS Output Format

```
data/output/
├── data/
│   ├── train/0.parquet       ← 80% of patients
│   ├── tuning/0.parquet      ← 10% of patients
│   └── held_out/0.parquet    ← 10% of patients
└── metadata/
    ├── codes.parquet         ← unique codes with descriptions
    ├── dataset.json          ← dataset metadata
    └── subject_splits.parquet
```

Each row in the data files follows the MEDS schema:

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | `int64` | Unique patient identifier |
| `time` | `timestamp[us]` | Timestamp of the event (`null` for static measurements) |
| `code` | `string` | Event code (e.g. `CHARTEVENT//001C_102//mmHg`, `MEDS_BIRTH`) |
| `numeric_value` | `float32` | Numeric value if applicable, otherwise `null` |

### Event types and codes

| Prefix | Source table | Example code |
|--------|-------------|--------------|
| `MEDS_BIRTH` | `syn_patients` | `MEDS_BIRTH` |
| `MEDS_DEATH` | `syn_patients`, `syn_admissions` | `MEDS_DEATH` |
| `GENDER` | `syn_patients` (static) | `GENDER//M` |
| `HOSPITAL_ADMISSION` | `syn_admissions` | `HOSPITAL_ADMISSION//Emergency department` |
| `HOSPITAL_DISCHARGE` | `syn_admissions` | `HOSPITAL_DISCHARGE//Home` |
| `ICU_ADMISSION` | `syn_icustays` | `ICU_ADMISSION//SICU` |
| `ICU_DISCHARGE` | `syn_icustays` | `ICU_DISCHARGE//RICU` |
| `CHARTEVENT` | `syn_chartevents` | `CHARTEVENT//001C_1021_25105//회/min` |
| `LAB` | `syn_labevents` | `LAB//001L3005//mg/dL` |
| `DIAGNOSIS` | `syn_diagnoses_icd` (static) | `DIAGNOSIS//KCD8//I251` |
| `PROCEDURE_ICD` | `syn_procedures_icd` | `PROCEDURE_ICD//ICD9CM//54.11` |
| `INPUT_START` | `syn_inputevents` | `INPUT_START//001I_1315//cc` |
| `OUTPUT` | `syn_outputevents` | `OUTPUT//001O_148//cc` |
| `MEDICATION` | `syn_emar` | `MEDICATION//12005122` |

### Dataset statistics (SYN-ICU)

| Metric | Value |
|--------|-------|
| Total events | 639,454 |
| Total patients | 1,328 |
| Static events (time = null) | 4,419 |
| Dynamic events | 635,035 |
| Events with numeric value | 605,941 |
| Unique codes | 182 |
| Train patients | 1,062 (80%) |
| Tuning patients | 132 (10%) |
| Held-out patients | 134 (10%) |

---

## Key Design Decisions

**UUID → int64 conversion:** K-MIMIC uses UUID strings for all identifiers. MEDS requires `int64`. We use SHA-256 hashing to produce stable, positive int64 values: the same UUID always maps to the same integer.

**Standalone pipeline:** MEDS-Extract CLI was evaluated but bypassed due to Windows/Hydra compatibility issues. The standalone `meds_convert.py` produces identical output with no external CLI dependency.

**Date-only timestamps:** `procedures_icd.chartdate` is a date without time. We add `23:59:59` to avoid temporal leakage in prediction tasks.

**Static events:** Diagnoses ICD and gender are treated as static (`time = null`) because no precise timestamp is available in the source data. They appear first in each patient's record as per the MEDS specification.

**Korean medical codes:** K-MIMIC uses the Korean Classification of Disease (KCD8) standard for diagnoses. These codes are preserved as-is in the MEDS output.

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

To be determined based on K-MIMIC data licensing terms.
