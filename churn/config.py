"""Central configuration: paths and settings used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SAMPLE_DATA_PATH = ROOT / "data" / "sample" / "synthetic_telco_churn.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Risk segmentation thresholds on predicted churn probability (configurable)
RISK_LOW_MAX = 0.30
RISK_HIGH_MIN = 0.60

# Illustrative business assumptions for "revenue at risk" (NOT measured results)
ASSUMED_RETENTION_MONTHS = 12   # months of revenue considered when sizing exposure
