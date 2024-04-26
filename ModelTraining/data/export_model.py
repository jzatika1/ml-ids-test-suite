import os
from joblib import dump
from tensorflow.keras.models import Model

def export_model(model, model_name, model_dir='../NetworkMonitor/models/'):
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
    model_path = os.path.join(model_dir, f"{model_name}.joblib")

    if isinstance(model, Model):
        # TensorFlow/Keras model
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(model_path)
    else:
        # Other models like xgboost, sklearn etc.
        dump(model, model_path)
    
    print(f"Saved {model_name} model to {model_path}.")