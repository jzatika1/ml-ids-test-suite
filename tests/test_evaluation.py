import numpy as np

from ModelTraining.evaluate_model.calculate_results import calculate_results


class ProbabilityModel:
    def predict_proba(self, X_test):
        return np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        )


def test_calculate_results_for_sklearn_style_model():
    metrics = calculate_results(
        ProbabilityModel(),
        np.asarray([[1], [2], [3]]),
        np.asarray([0, 1, 0]),
    )

    assert metrics["Accuracy"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["F1 Score"] == 1.0
    assert metrics["ROC-AUC Score"] == 1.0
