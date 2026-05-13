from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "DataPreprocessing" / "preprocessed_data"


def load_data(data_dir=DEFAULT_DATA_DIR):
    data_path = Path(data_dir)
    expected_files = {
        "X_train": data_path / "X_train_stratify.csv",
        "X_test": data_path / "X_test_stratify.csv",
        "y_train": data_path / "y_train_encoded_stratify.csv",
        "y_test": data_path / "y_test_encoded_stratify.csv",
    }
    missing = [str(path) for path in expected_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Preprocessed training data is missing. Run DataPreprocessing/main.py first. "
            f"Missing: {', '.join(missing)}"
        )

    X_train = pd.read_csv(expected_files["X_train"])
    X_test = pd.read_csv(expected_files["X_test"])
    y_train_encoded = np.loadtxt(expected_files["y_train"], delimiter=",", dtype=int)
    y_test_encoded = np.loadtxt(expected_files["y_test"], delimiter=",", dtype=int)

    return X_train, X_test, y_train_encoded, y_test_encoded
