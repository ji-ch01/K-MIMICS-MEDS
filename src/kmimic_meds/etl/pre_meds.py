"""
Pre-MEDS transformation step for K-MIMIC.

Responsibilities:
- Load raw K-MIMIC source tables
- Apply timestamp resolution (date-only columns → full datetime)
- Compute derived columns (e.g. year_of_birth from age)
- Resolve duplicate event sources across tables
- Write cleaned Parquet files to intermediate_dir

To be completed after K-MIMIC schema exploration.
"""

import argparse
from pathlib import Path

import pandas as pd


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: load K-MIMIC tables once schema is known
    # Example:
    # patients = pd.read_csv(input_dir / "patients.csv")
    # patients = transform_patients(patients)
    # patients.to_parquet(output_dir / "patients.parquet", index=False)

    print(f"Pre-MEDS: reading from {input_dir}")
    print(f"Pre-MEDS: writing to   {output_dir}")
    print("TODO: implement transformations after K-MIMIC schema is known.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-MEDS transformation for K-MIMIC")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
