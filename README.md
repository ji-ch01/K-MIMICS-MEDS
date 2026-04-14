# K-MIMIC-MEDS

ETL pipeline to convert the **Synthetic K-MIMIC** (Korean ICU) dataset into the **MEDS** (Medical Event Data Standard) format.

This project is part of a research initiative to extend the MEDS ecosystem to non-English clinical datasets, starting with the Korean ICU synthetic dataset published by KHDP.

---

## Overview

[MEDS](https://medical-event-data-standard.github.io/) is a minimal, interoperable data standard for longitudinal medical event data, designed for reproducible machine learning research in healthcare.

This repository contains:
- A **Pre-MEDS** transformation step to clean and reshape raw K-MIMIC tables
- A **MESSY configuration file** (`configs/messy.yaml`) describing the event mapping
- A **MEDS-Extract-based pipeline** to produce the final MEDS-compliant Parquet dataset
- The resulting **Synthetic K-MIMIC-MEDS dataset**

The ETL code is packaged as a Python package (`kmimic_meds`) intended for publication on PyPI.

---

## Data Sources

| Source | Description |
|--------|-------------|
| [Synthetic K-MIMIC (SYN-ICU)](https://khdp.net/database/data-search-detail/SYN-ICU) | Synthetic Korean ICU dataset published by KHDP |
| [MIMIC-IV-MEDS](https://physionet.org/content/mimic-iv-demo-meds/0.0.1/) | Reference MEDS conversion of MIMIC-IV (used as model) |

> Raw K-MIMIC data is **not included** in this repository. Download it from the KHDP portal and place it under `data/raw/`.

---

## Project Structure

```
K-MIMIC-MEDS/
├── configs/
│   └── messy.yaml              # MEDS-Extract event mapping configuration
├── data/
│   ├── raw/                    # Raw K-MIMIC source files (not versioned)
│   ├── intermediate/           # Pre-MEDS transformed Parquet files
│   └── output/                 # Final MEDS-compliant dataset
├── docs/                       # Additional documentation and notes
├── src/
│   └── kmimic_meds/
│       ├── __init__.py
│       ├── etl/
│       │   ├── __init__.py
│       │   └── pre_meds.py     # Pre-MEDS transformation logic
│       └── utils/
│           ├── __init__.py
│           └── io.py           # I/O helpers
├── tests/
│   └── test_pre_meds.py
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/K-MIMIC-MEDS.git
cd K-MIMIC-MEDS
pip install -e ".[dev]"
```

---

## Usage

### Step 1 — Place raw data

Download the Synthetic K-MIMIC dataset from [KHDP](https://khdp.net/database/data-search-detail/SYN-ICU) and place the files under `data/raw/`.

### Step 2 — Run Pre-MEDS

```bash
python -m kmimic_meds.etl.pre_meds \
    --input_dir data/raw \
    --output_dir data/intermediate
```

### Step 3 — Run MEDS-Extract

```bash
MEDS_transform-pipeline \
    pkg://MEDS_extract.configs._extract.yaml \
    --overrides \
    input_dir=data/intermediate \
    output_dir=data/output \
    event_conversion_config_fp=configs/messy.yaml \
    dataset.name=K-MIMIC-MEDS \
    dataset.version=0.1.0
```

---

## MEDS Output Format

The output follows the standard MEDS directory layout:

```
data/output/
├── data/
│   ├── train/
│   │   └── 0.parquet
│   ├── tuning/
│   │   └── 0.parquet
│   └── held_out/
│       └── 0.parquet
└── metadata/
    ├── codes.parquet
    ├── dataset.json
    └── subject_splits.parquet
```

Each row in the data files follows the MEDS schema:

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | `int64` | Unique patient identifier |
| `time` | `timestamp[us]` | Timestamp of the event (`null` for static measurements) |
| `code` | `string` | Event code (e.g. `VITALS//HeartRate`, `MEDS_BIRTH`) |
| `numeric_value` | `float32` | Numeric value if applicable |

---

## References

- [MEDS Documentation](https://medical-event-data-standard.github.io/)
- [MEDS GitHub](https://github.com/Medical-Event-Data-Standard/meds)
- [MEDS-Extract](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS)
- [MEDS-Transforms](https://github.com/mmcdermott/MEDS_transforms)
- [KHDP SYN-ICU Dataset](https://khdp.net/database/data-search-detail/SYN-ICU)

---

## License

To be determined based on K-MIMIC data licensing terms.
