"""Risk segmentation and an illustrative (assumption-based) revenue-exposure estimate."""
import pandas as pd

from . import config


def risk_tier(prob: float) -> str:
    if prob >= config.RISK_HIGH_MIN:
        return "High"
    if prob >= config.RISK_LOW_MAX:
        return "Medium"
    return "Low"


def segment(df: pd.DataFrame, prob_col: str = "churn_probability",
            months: int = config.ASSUMED_RETENTION_MONTHS) -> pd.DataFrame:
    """Summarise customers per tier. `assumed_revenue_exposure` = sum(MonthlyCharges) * months
    for the tier -- a configurable illustration, NOT a measured or guaranteed figure."""
    d = df.copy()
    d["risk_tier"] = d[prob_col].map(risk_tier)
    out = d.groupby("risk_tier").agg(
        customers=(prob_col, "size"),
        avg_predicted_probability=(prob_col, "mean"),
        actual_churn_rate=(config.TARGET, "mean") if config.TARGET in d else (prob_col, "mean"),
        monthly_revenue=("MonthlyCharges", "sum"),
    )
    out["assumed_revenue_exposure"] = out["monthly_revenue"] * months
    return out.reindex(["High", "Medium", "Low"]).dropna(how="all").round(3)
