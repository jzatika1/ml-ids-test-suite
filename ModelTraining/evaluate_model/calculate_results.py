import numpy as np
import xgboost as xgb
import cudf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Model

def calculate_results(model, X_test, y_test_encoded):
    # Prepare X_test and predictions based on the model type
    if isinstance(model, (xgb.Booster, xgb.XGBModel)):
        # Handle xgboost models
        if not isinstance(X_test, xgb.DMatrix):
            dtest = xgb.DMatrix(X_test)
        else:
            dtest = X_test
        preds_proba = model.predict(dtest)
    elif isinstance(model, Model):
        # Handle TensorFlow/Keras models
        if isinstance(X_test, cudf.DataFrame):
            X_test = X_test.to_pandas().values
        preds_proba = model.predict(X_test)
    else:
        # Handle cuML and potentially sklearn models
        if isinstance(X_test, (cudf.DataFrame, xgb.DMatrix)):
            raise ValueError("X_test must be a NumPy array or pandas DataFrame for cuML or sklearn models.")
        if isinstance(X_test, np.ndarray):
            X_test = cudf.DataFrame(X_test)
        preds_proba = model.predict_proba(X_test)
    
    # Prepare y_test for metrics calculation
    y_test_labels = np.argmax(y_test_encoded, axis=1) if len(y_test_encoded.shape) > 1 else y_test_encoded.squeeze()
    
    # Initialize ROC-AUC score and calculate if probabilities are available
    roc_auc = 'Not computed'
    try:
        if preds_proba.ndim == 2:
            roc_auc = roc_auc_score(y_test_labels, preds_proba, multi_class='ovr')
        roc_auc = f"{roc_auc:.2f}" if isinstance(roc_auc, float) else roc_auc
    except Exception as e:
        print(f"Error computing ROC-AUC: {e}")

    # Calculate class labels from probabilities if necessary
    preds_classes = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 else preds_proba.round()

    # Calculate other metrics
    metrics = {
        'Accuracy': accuracy_score(y_test_labels, preds_classes),
        'Precision': precision_score(y_test_labels, preds_classes, average='macro', zero_division=1),
        'Recall': recall_score(y_test_labels, preds_classes, average='macro', zero_division=1),
        'F1 Score': f1_score(y_test_labels, preds_classes, average='macro', zero_division=1),
        'ROC-AUC Score': roc_auc
    }

    return metrics