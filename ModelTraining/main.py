import configparser
import logging
from pathlib import Path

try:
    from .data.data_processor import load_data
    from .data.export_model import export_model
    from .evaluate_model.calculate_results import calculate_results
    from .models import neural_network_model, random_forest_model, svm_model, xgboost_model
except ImportError:  # pragma: no cover - supports `python ModelTraining/main.py`
    from data.data_processor import load_data
    from data.export_model import export_model
    from evaluate_model.calculate_results import calculate_results
    from models import neural_network_model, random_forest_model, svm_model, xgboost_model

logger = logging.getLogger(__name__)


def main():
    config_path = Path(__file__).resolve().parent / "config" / "config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)

    X_train, X_test, y_train, y_test = load_data()

    model_functions = {
        "xgboost": xgboost_model.train_xgboost,
        "random_forest": random_forest_model.train_random_forest,
        "neural_network": neural_network_model.train_neural_network,
        "svm": svm_model.train_svm,
    }

    results = {}

    for model_name, train_func in model_functions.items():
        if config.getboolean("models", model_name):
            logger.info("Training %s...", model_name)
            trained_model = train_func(X_train, y_train)
            metrics = calculate_results(trained_model, X_test, y_test)
            results[model_name] = metrics

            logger.info("Evaluation metrics for %s model:", model_name)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    formatted_value = (
                        f"{value * 100:.3f}%" if metric != "ROC-AUC Score" else f"{value:.3f}"
                    )
                else:
                    formatted_value = value
                logger.info("  %s: %s", metric, formatted_value)

            export_model(trained_model, model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
