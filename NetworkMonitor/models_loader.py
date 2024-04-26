from joblib import load
import os
import logging

def load_models(models_directory):
    """
    Load all machine learning models from the specified directory.

    Parameters:
    - models_directory: str, the path to the directory containing the model files.

    Returns:
    - models: dict, a dictionary mapping model names to their loaded instances.
    """
    models = {}
    if not os.path.isdir(models_directory):
        logging.error(f"Models directory does not exist: {models_directory}")
        return models

    for filename in os.listdir(models_directory):
        if filename.endswith(".joblib") or filename.endswith(".pkl"):
            model_path = os.path.join(models_directory, filename)
            try:
                model_name = filename.rsplit('.', 1)[0]  # Remove the file extension to get the model name
                models[model_name] = load(model_path)
                logging.info(f"{model_name} model loaded successfully from {model_path}.")
            except Exception as e:
                logging.error(f"Failed to load model {filename} due to an error: {e}")
    return models