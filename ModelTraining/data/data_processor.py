import os
import numpy as np
import pandas as pd

def load_data():
    # Relative path to your mappings directory
    saved_data_path = '../DataPreprocessing/preprocessed_data/'

    # Get the absolute path to the mappings directory
    absolute_data_path = os.path.abspath(saved_data_path)

    # Load X_train and X_test from CSV files
    X_train = pd.read_csv(os.path.join(absolute_data_path, "X_train_stratify.csv"))
    X_test = pd.read_csv(os.path.join(absolute_data_path, "X_test_stratify.csv"))

    # Load y_train_encoded and y_test_encoded from CSV files
    y_train_encoded = np.loadtxt(os.path.join(absolute_data_path, "y_train_encoded_stratify.csv"), delimiter=",", dtype=int)
    y_test_encoded = np.loadtxt(os.path.join(absolute_data_path, "y_test_encoded_stratify.csv"), delimiter=",", dtype=int)

    return X_train, X_test, y_train_encoded, y_test_encoded