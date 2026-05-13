from pathlib import Path

import xgboost as xgb
from joblib import dump

try:
    from tensorflow.keras.models import Model as KerasModel
except ImportError:  # pragma: no cover - optional deep-learning dependency
    KerasModel = None


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "NetworkMonitor" / "models"


def export_model(model, model_name, model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, xgb.Booster):
        model_path = model_dir / f"{model_name}.json"
        model.save_model(str(model_path))
    elif KerasModel is not None and isinstance(model, KerasModel):
        model_path = model_dir / f"{model_name}.keras"
        model.save(model_path)
    else:
        model_path = model_dir / f"{model_name}.joblib"
        dump(model, model_path)

    print(f"Saved {model_name} model to {model_path}.")
    return model_path
