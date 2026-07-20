from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

COLUMNS_PATH = BASE_DIR / "artifacts" / "training_columns.pkl"