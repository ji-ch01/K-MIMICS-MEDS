"""I/O helpers for reading K-MIMIC source files."""

from pathlib import Path

import pandas as pd


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    elif suffix in (".csv", ".gz"):
        return pd.read_csv(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
