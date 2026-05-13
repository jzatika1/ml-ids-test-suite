import logging

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import cudf
except ImportError:  # pragma: no cover - optional GPU dependency
    cudf = None

try:
    from tensorflow.keras.models import Model as KerasModel
except ImportError:  # pragma: no cover - optional deep-learning dependency
    KerasModel = None


logger = logging.getLogger(__name__)


def _is_cudf_frame(value):
    return cudf is not None and isinstance(value, cudf.DataFrame)


def _to_model_input(X_test):
    if _is_cudf_frame(X_test):
        return X_test.to_pandas()
    return X_test


def _predict_probabilities(model, X_test):
    if isinstance(model, xgb.Booster):
        dtest = X_test if isinstance(X_test, xgb.DMatrix) else xgb.DMatrix(X_test)
        return model.predict(dtest)

    if isinstance(model, xgb.XGBModel):
        model_input = _to_model_input(X_test)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(model_input)
        return model.predict(model_input)

    if KerasModel is not None and isinstance(model, KerasModel):
        return model.predict(_to_model_input(X_test))

    if isinstance(X_test, xgb.DMatrix):
        raise ValueError("X_test must be array-like or a DataFrame for non-XGBoost models.")

    model_input = _to_model_input(X_test)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(model_input)
    return model.predict(model_input)


def _labels_from_target(y_test_encoded):
    target = np.asarray(y_test_encoded)
    return np.argmax(target, axis=1) if target.ndim > 1 else target.squeeze()


def _classes_from_predictions(predictions):
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:
        return np.argmax(predictions, axis=1)
    return predictions.round().astype(int)


def _roc_auc(y_true, predictions):
    predictions = np.asarray(predictions)
    try:
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            return roc_auc_score(y_true, predictions[:, 1])
        if predictions.ndim == 2 and predictions.shape[1] > 2:
            return roc_auc_score(y_true, predictions, multi_class="ovr")
    except ValueError as exc:
        logger.warning("ROC-AUC could not be computed: %s", exc)
    return "Not computed"


def calculate_results(model, X_test, y_test_encoded):
    predictions = _predict_probabilities(model, X_test)
    y_test_labels = _labels_from_target(y_test_encoded)
    predicted_classes = _classes_from_predictions(predictions)

    return {
        "Accuracy": accuracy_score(y_test_labels, predicted_classes),
        "Precision": precision_score(
            y_test_labels,
            predicted_classes,
            average="macro",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test_labels,
            predicted_classes,
            average="macro",
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test_labels,
            predicted_classes,
            average="macro",
            zero_division=0,
        ),
        "ROC-AUC Score": _roc_auc(y_test_labels, predictions),
    }
