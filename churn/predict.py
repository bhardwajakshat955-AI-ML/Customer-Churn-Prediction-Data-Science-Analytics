"""Load the saved pipeline and score customers (same transformations as training)."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config
from .risk import risk_tier


class ModelNotFoundError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model():
    if not config.MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"Model artifact not found at {config.MODEL_PATH}. Train it first: python -m churn.train")
    return joblib.load(config.MODEL_PATH)


def load_metadata() -> dict:
    return json.loads(config.METADATA_PATH.read_text()) if config.METADATA_PATH.exists() else {}


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[config.FEATURES]  # enforce training column order
    proba = load_model().predict_proba(X)[:, 1]
    thr = load_metadata().get("threshold", 0.5)
    return pd.DataFrame({
        "churn_probability": proba.round(4),
        "churn_prediction": (proba >= thr).astype(int),
        "risk_tier": [risk_tier(p) for p in proba],
    }, index=df.index)


def predict_one(record: dict) -> dict:
    row = predict_dataframe(pd.DataFrame([record])).iloc[0]
    return {"churn_probability": float(row.churn_probability),
            "churn_prediction": bool(row.churn_prediction), "risk_tier": row.risk_tier}
