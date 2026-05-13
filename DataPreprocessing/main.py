import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_loading import load_files
    from .data_splitting import split_data
    from .label_encoding import encode_labels
except ImportError:  # pragma: no cover - supports `python DataPreprocessing/main.py`
    from data_loading import load_files
    from data_splitting import split_data
    from label_encoding import encode_labels

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PREPROCESSED_DIR = BASE_DIR / "preprocessed_data"
MAPPINGS_DIR = BASE_DIR / "model_mappings"


def collect_dataset_paths(dataset_names):
    filepaths = []
    for dataset_name in dataset_names:
        dataset_dir = DATA_DIR / dataset_name
        filepaths.extend(glob.glob(str(dataset_dir / "*.csv")))
        filepaths.extend(glob.glob(str(dataset_dir / "*.xlsx")))
    return sorted(filepaths)


def main(dataset_names=None):
    pd.set_option("display.float_format", "{:.2f}".format)

    dataset_names = dataset_names or [
        "UNSW-NB15",
        # "TON_IoT",
        # "CICIDS17",
        # "ROUTESMART",
    ]
    filepaths = collect_dataset_paths(dataset_names)
    if not filepaths:
        raise FileNotFoundError(
            f"No CSV or XLSX files found in {DATA_DIR}. Add datasets before preprocessing."
        )

    X_combined, y_combined, label_bytes = load_files(filepaths)
    logger.info("Label byte distribution:\n%s", label_bytes)

    X_train, X_test, y_train, y_test = split_data(X_combined, y_combined)
    y_train_encoded, y_test_encoded = encode_labels(
        y_train,
        y_test,
        save_folder=MAPPINGS_DIR,
    )

    logger.info("X_train shape: %s", X_train.shape)
    logger.info("y_train shape: %s", y_train.shape)
    logger.info("X_test shape: %s", X_test.shape)
    logger.info("y_test shape: %s", y_test.shape)

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PREPROCESSED_DIR / "X_train_stratify.csv", index=False)
    X_test.to_csv(PREPROCESSED_DIR / "X_test_stratify.csv", index=False)
    np.savetxt(
        PREPROCESSED_DIR / "y_train_encoded_stratify.csv",
        y_train_encoded,
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        PREPROCESSED_DIR / "y_test_encoded_stratify.csv",
        y_test_encoded,
        delimiter=",",
        fmt="%d",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
