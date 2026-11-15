"""Paths, column names and constants used by every module.

Keeping them in one file means a column list or a path is written once.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "telco_customer_churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

METADATA_FILE = "model_metadata.json"

# ------------------------------------------------------------ constants
RANDOM_SEED = 42
MODEL_THREADS = 4          # threads used by LightGBM for training
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15

# Precision level used for the "recall at precision" metric.
# Meaning: at least 6 of every 10 customers we contact are real churners.
TARGET_PRECISION = 0.60

# -------------------------------------------------------------- columns
ID_COL = "customerID"
TARGET_COL = "Churn"

# Numeric columns exactly as they arrive in the raw data.
RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

# Categorical columns exactly as they arrive in the raw data.
RAW_CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

# Optional services a customer can add on top of the internet plan.
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# One engineered feature: how many add-on services the customer has.
ENGINEERED_NUMERIC = ["num_addon_services"]

# Final lists that go into the model.
MODEL_NUMERIC = RAW_NUMERIC + ENGINEERED_NUMERIC
MODEL_CATEGORICAL = RAW_CATEGORICAL
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL
