"""Data loading, validation and cleaning."""
import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [config.ID_COLUMN, config.TARGET, *config.FEATURES]


def resolve_data_path(use_sample: bool = False) -> Path:
    """Return the dataset path; fail with clear instructions if it is missing."""
    if use_sample:
        if not config.SAMPLE_DATA_PATH.exists():
            from .sample_data import write_sample
            write_sample(config.SAMPLE_DATA_PATH)
        log.warning("Using SYNTHETIC sample data. Results are not real-world results.")
        return config.SAMPLE_DATA_PATH
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.RAW_DATA_PATH}.\n"
            "Download the IBM Telco Customer Churn CSV (see data/README.md) and place it there, "
            "or run with --use-sample for a synthetic demo dataset."
        )
    return config.RAW_DATA_PATH


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco data. Stateless (no statistics learned), so it is leakage-safe.

    - TotalCharges has blank strings for new customers -> NaN (imputed later in the pipeline)
    - SeniorCitizen 0/1 -> No/Yes so it is treated as categorical
    - Churn Yes/No -> 1/0
    - Duplicate customer rows are dropped
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0})
    if df[config.TARGET].isna().any():
        raise ValueError("Churn column contains values other than Yes/No.")
    before = len(df)
    df = df.drop_duplicates(subset=config.ID_COLUMN)
    if len(df) < before:
        log.info("Dropped %d duplicate customer rows", before - len(df))
    return df.reset_index(drop=True)


def load_clean(use_sample: bool = False) -> tuple[pd.DataFrame, Path]:
    path = resolve_data_path(use_sample)
    return clean(load_raw(path)), path
