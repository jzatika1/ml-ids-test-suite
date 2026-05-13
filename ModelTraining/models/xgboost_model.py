import os

import xgboost as xgb

try:
    from ..data import get_num_classes
except ImportError:  # pragma: no cover - supports direct script execution
    from data import get_num_classes


def train_xgboost(
    X_train,
    y_train,
    num_classes=None,
    learning_rate=0.25,
    max_depth=23,
    subsample=0.8,
    num_boost_round=200,
    seed=42,
):
    num_classes = num_classes or get_num_classes.num_classes()
    if not num_classes:
        raise ValueError(
            "Unable to determine class count. Run DataPreprocessing/main.py first "
            "or pass num_classes explicitly."
        )

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "tree_method": os.getenv("XGBOOST_TREE_METHOD", "hist"),
        "device": os.getenv("XGBOOST_DEVICE", "cpu"),
        "seed": seed,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    return model
