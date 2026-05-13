import logging
from pathlib import Path

import xgboost as xgb
from joblib import load

logger = logging.getLogger(__name__)


def load_models(models_directory):
    """
    Load all machine learning models from the specified directory.

    Parameters:
    - models_directory: str, the path to the directory containing the model files.

    Returns:
    - models: dict, a dictionary mapping model names to their loaded instances.
    """
    models = {}
    models_path = Path(models_directory)
    if not models_path.is_dir():
        logger.error("Models directory does not exist: %s", models_path)
        return models

    for model_path in sorted(models_path.iterdir()):
        if model_path.suffix not in {".joblib", ".json", ".pkl"}:
            continue

        try:
            model_name = model_path.stem
            if model_path.suffix == ".json":
                booster = xgb.Booster()
                booster.load_model(str(model_path))
                models[model_name] = booster
            else:
                models[model_name] = load(str(model_path))
            logger.info("%s model loaded successfully from %s.", model_name, model_path)
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_path.name, exc)
    return models
