from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Data paths
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"

CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_customer_churn.csv"


# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"

PLOTS_DIR = OUTPUT_DIR / "plots"

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


# Plot size
FIG_SIZE = (8, 5)