# Standard library imports
import os
import random
import signal
import sys
import time
import json
from multiprocessing import Manager
from queue import Empty

# Third-party library imports
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from joblib import parallel_backend
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

# Dask imports for parallel computing
import dask.array as da
import dask.dataframe as dd

n_gpu = 8
study_name = 'XGBoost_Hyperparameter_Optimization'

hyperparameter_config = {
    "max_depth": {
        "type": "int",
        "range": (1, 100),
        "enabled": True
    },
    "num_boost_round": {
        "type": "int",
        "range": (1, 15),
        "enabled": False
    },
    "learning_rate": {
        "type": "float",
        "range": (0.01, 0.99),
        "enabled": True
    }
}

def find_unique_parameters(trial):
    # Gather parameters from all trials, including those that are running
    used_params = set()
    for t in trial.study.get_trials(deepcopy=False):
        if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING]:
            params_hash = frozenset(t.params.items())
            used_params.add(params_hash)

    def suggest_new_parameters():
        while True:
            suggested_params = {}
            for param, info in hyperparameter_config.items():
                if info["enabled"]:
                    if info["type"] == "int":
                        suggested_params[param] = trial.suggest_int(param, *info["range"])
                    elif info["type"] == "float":
                        suggested_params[param] = trial.suggest_float(param, *info["range"])
            params_hash = frozenset(suggested_params.items())
            if params_hash not in used_params:
                break  # Found a new unique combination
        return suggested_params

    return suggest_new_parameters()


def objective(study, trial, X_train, X_test, y_train, y_test, gpu_queue):
    """
    Objective function for hyperparameter optimization with Optuna and XGBoost.

    Manages GPU resource allocation to ensure efficient training and handles
    retry logic if GPUs are not immediately available. This function is designed
    to be used within an Optuna optimization process to evaluate the performance
    of different sets of hyperparameters on the given dataset.

    Parameters:
    - study: Optuna study object tracking the optimization process.
    - trial: Optuna trial object for generating hyperparameter suggestions.
    - X_train, X_test: Training and test feature sets.
    - y_train, y_test: Training and test labels.
    - gpu_queue: Queue for managing GPU resource allocation.

    Returns:
    - The evaluation metric (e.g., F1 score) of the model trained with the trial's hyperparameters.
    """

    # Initial setup for retry logic in case the GPU queue is empty
    max_retries = 3
    retry_delay = 3  # Delay in seconds between retries

    # Attempt to get a GPU ID from the queue with retries
    for attempt in range(max_retries):
        try:
            gpu_id = gpu_queue.get(timeout=10)  # Try to get GPU ID with a 10-second timeout
            print(f"Retrieved GPU ID: {gpu_id}")
            break  # Exit loop if successful
        except Empty as e:  # If queue is empty, retry after a delay
            if attempt < max_retries - 1:  # Check if more retries are allowed
                print(f"Retry {attempt + 1}/{max_retries} - GPU queue is empty. Retrying in {retry_delay} seconds.")
                time.sleep(retry_delay)
            else:
                print(f"Maximum retries exceeded. GPU queue is empty.")
                raise  # Raise exception if maximum retries are exceeded
        except EOFError as e:  # Handle unexpected EOFError when accessing the queue
            print(f"EOFError occurred while retrieving GPU ID from the queue. Retrying...")
            time.sleep(retry_delay)
            if attempt == max_retries - 1:
                print(f"Maximum retries exceeded. EOFError persists.")
                raise  # Raise exception if EOFError persists after retries
    
    trials_df = study.trials_dataframe()
    
    # Check for duplicate trials to avoid redundant computation
    if not trials_df.empty:
        optim_params = find_unique_parameters(trial)
        
    # Path to your mappings file
    mappings_file_path = '../DataPreprocessing/model_mappings/mappings.json'
    
    # Load mappings file
    with open(mappings_file_path, 'r') as f:
        mappings = json.load(f)
    
    # Determine the number of unique classes from the mappings
    num_classes = len(mappings)
        
    try:
        # Setup fixed hyperparameters
        fixed_params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'tree_method': 'hist',
            'device': f'cuda:{gpu_id}',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'seed': 42,
        }
    
        # Merge the optimized and fixed hyperparameters
        training_params = {**optim_params, **fixed_params}
    
        # Prepare data for xgboost
        dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
        dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)
    
        # Determine if 'num_boost_round' is to be used
        use_num_boost_round = hyperparameter_config.get('num_boost_round', {}).get('enabled', False)
        
        # Train the model, conditionally including 'num_boost_round'
        if use_num_boost_round and 'num_boost_round' in optim_params:
            num_boost_round = optim_params.pop('num_boost_round')
            model = xgb.train(training_params, dtrain, num_boost_round=num_boost_round)
        else:
            # 'num_boost_round' is either disabled or not set; proceed without specifying it
            model = xgb.train(training_params, dtrain)
    
        # Make predictions and evaluate
        preds = model.predict(dtest)
        preds_classes = preds.argmax(axis=1)
        f1 = f1_score(y_test, preds_classes, average='macro')
        accuracy = accuracy_score(y_test, preds_classes)
    finally:
        # Return GPU ID to the queue after use
        gpu_queue.put(gpu_id)
        print(f"Returned GPU ID: {gpu_id}")

        # Return the evaluation metrics
        return f1, accuracy

def main():
    """
    Main function to execute the hyperparameter optimization process for an XGBoost model using Optuna,
    leveraging Dask for parallel computation to handle large datasets efficiently.

    The process involves:
    1. Loading pre-processed training and testing datasets from Parquet and CSV files.
    2. Converting Dask dataframes and arrays to their NumPy counterparts for model training.
    3. Setting up an Optuna study to maximize model performance metrics, utilizing an SQLite database to store trial results.
    4. Executing the optimization across multiple GPUs in parallel, dynamically allocating GPUs to trials.
    5. Handling KeyboardInterrupts gracefully, saving progress before exiting to ensure no optimization progress is lost.

    The optimization targets are specified as maximizing objectives, and the results, including the best trial's
    parameters and objective values, are printed upon completion. This script demonstrates a scalable approach to
    machine learning model tuning in a distributed computing environment.
    
    Note:
    - Ensure the availability of the GPU resources and the proper configuration of the Dask environment for parallel processing.
    - Adjust the `n_gpu`, `study_name`, and `save_dir` variables as needed to fit the execution environment.
    """
    
    try:
        # Directory where the parsed data is saved
        save_dir = '../DataPreprocessing/preprocessed_data/'
    
        # Load X_train and X_test from CSV files
        X_train = pd.read_csv(f"{save_dir}X_train_stratify.csv")
        X_test = pd.read_csv(f"{save_dir}X_test_stratify.csv")
    
        # Load y_train_encoded and y_test_encoded from CSV files
        y_train_encoded = np.loadtxt(f"{save_dir}y_train_encoded_stratify.csv", delimiter=",", dtype=int)
        y_test_encoded = np.loadtxt(f"{save_dir}y_test_encoded_stratify.csv", delimiter=",", dtype=int)
        
        # Define the name of the SQLite DB file
        sqlite_db_name = "database.db"
        
        # Create or load an Optuna study with SQLite backend
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f'sqlite:///{sqlite_db_name}',
            load_if_exists=True  # This will load the study if it already exists, preventing errors on re-runs
        )
        
        # Optimize with Optuna
        study = optuna.create_study(directions=['maximize', 'maximize'], study_name=study_name)
        with Manager() as manager:
            # Initialize the queue by adding available GPU IDs.
            gpu_queue = manager.Queue()
            for i in range(n_gpu):
                gpu_queue.put(i)
            
            with parallel_backend("multiprocessing", n_jobs=n_gpu):
                study.optimize(
                    lambda trial: objective(study, trial, X_train, X_test, y_train_encoded, y_test_encoded, gpu_queue),
                    n_trials=100,
                    n_jobs=n_gpu,
                )
        
        print("====DONE====")
        
        best_trials = study.best_trials
        if len(best_trials) > 0:
            trial = best_trials[0]
            print(f"Best trial (Trial {trial.number}):")
            print(" Objective values:")
            for i, value in enumerate(trial.values, 1):
                print(f" Objective {i}: {value}")
            print(" Parameter values:")
            for name, value in trial.params.items():
                print(f" {name}: {value}")
        else:
            print("No best trials found.")
    
    except KeyboardInterrupt:
        try:
            logging.info("Process interrupted. Attempting to save the current study...")
            
            # Cleanup code
            study.trials_dataframe().to_csv("latest_study_results.csv")
            logging.info("Progress saved. Exiting now.")
        except KeyboardInterrupt:
            logging.warning("Additional interrupt received during cleanup. Exiting immediately.")
        finally:
            sys.exit(0)

if __name__ == '__main__':
    main()