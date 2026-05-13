import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "DataPreprocessing" / "preprocessed_data"
DEFAULT_MAPPINGS_PATH = REPO_ROOT / "DataPreprocessing" / "model_mappings" / "mappings.json"


def load_preprocessed_data(data_dir=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    X_train = pd.read_csv(data_dir / "X_train_stratify.csv")
    X_test = pd.read_csv(data_dir / "X_test_stratify.csv")
    y_train = np.loadtxt(data_dir / "y_train_encoded_stratify.csv", delimiter=",", dtype=int)
    y_test = np.loadtxt(data_dir / "y_test_encoded_stratify.csv", delimiter=",", dtype=int)
    return X_train, X_test, y_train, y_test


def load_num_classes(mappings_path=DEFAULT_MAPPINGS_PATH):
    with Path(mappings_path).open("r", encoding="utf-8") as file:
        return len(json.load(file))


def suggest_parameters(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "num_boost_round": trial.suggest_int("num_boost_round", 50, 300),
    }


def objective(trial, X_train, X_test, y_train, y_test, num_classes, device):
    trial_params = suggest_parameters(trial)
    num_boost_round = trial_params.pop("num_boost_round")
    params = {
        **trial_params,
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": device,
        "eval_metric": "mlogloss",
        "seed": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    predictions = model.predict(dtest)
    predicted_classes = predictions.argmax(axis=1)
    return (
        f1_score(y_test, predicted_classes, average="macro"),
        accuracy_score(y_test, predicted_classes),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize XGBoost hyperparameters with Optuna.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--mappings-path", default=DEFAULT_MAPPINGS_PATH, type=Path)
    parser.add_argument("--n-trials", default=100, type=int)
    parser.add_argument("--n-jobs", default=1, type=int)
    parser.add_argument("--device", default="cpu", help="XGBoost device, for example cpu or cuda.")
    parser.add_argument("--study-name", default="xgboost_hyperparameter_optimization")
    parser.add_argument("--storage", default="sqlite:///optuna_xgboost.db")
    return parser.parse_args()


def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_preprocessed_data(args.data_dir)
    num_classes = load_num_classes(args.mappings_path)

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            X_test,
            y_train,
            y_test,
            num_classes,
            args.device,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    logger.info("Optimization complete.")
    for trial in study.best_trials:
        logger.info("Best trial %s values=%s params=%s", trial.number, trial.values, trial.params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
