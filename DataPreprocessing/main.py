import glob
import os
import numpy as np
import pandas as pd
from data_loading import load_files
from data_splitting import split_data
from label_encoding import encode_labels

# Set display options
pd.set_option('display.float_format', '{:.2f}'.format)  # Adjust format as needed
pd.set_option('display.max_rows', None)  # Ensures all rows are displayed

if __name__ == "__main__":
    base_dir = 'data/'
    directories = [
        'UNSW-NB15',
        #'TON_IoT',
        #'CICIDS17',
        #'ROUTESMART'
    ]

    filepaths = []
    for directory in directories:
        csv_pattern = os.path.join(base_dir, directory, "*.csv")
        xlsx_pattern = os.path.join(base_dir, directory, "*.xlsx")
        filepaths.extend(glob.glob(csv_pattern))
        filepaths.extend(glob.glob(xlsx_pattern))

    X_combined, y_combined, label_bytes = load_files(filepaths)

    print(label_bytes)

    X_train, X_test, y_train, y_test = split_data(X_combined, y_combined)
    y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    save_dir = 'preprocessed_data'
    os.makedirs(save_dir, exist_ok=True)

    # Save data to CSV files
    X_train.to_csv(os.path.join(save_dir, "X_train_stratify.csv"), index=False)
    X_test.to_csv(os.path.join(save_dir, "X_test_stratify.csv"), index=False)

    # Saving y_train_encoded to CSV
    np.savetxt(os.path.join(save_dir, "y_train_encoded_stratify.csv"), y_train_encoded, delimiter=",", fmt='%d', header='')
    
    # Saving y_test_encoded to CSV
    np.savetxt(os.path.join(save_dir, "y_test_encoded_stratify.csv"), y_test_encoded, delimiter=",", fmt='%d', header='')