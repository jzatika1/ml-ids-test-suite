# ML IDS Test Suite

Machine-learning intrusion detection research code presented at the **2024 IEEE/ACM 17th International Conference on Utility and Cloud Computing (UCC 2024)** in the paper:

**Improving IDS Performance with XGBoost Hyperparameter Optimization and Real-Time Analysis**  
Anthony Zatika and Joel Coffman  
DOI: `10.1109/UCC63386.2024.00037`

This repository focuses on practical network IDS experimentation: dataset normalization, XGBoost training, hyperparameter optimization, and realtime Zeek `conn.log` analysis.

## What This Repo Does

- Normalizes IDS datasets into a compact shared feature contract.
- Trains XGBoost, Random Forest, SVM, and optional TensorFlow baselines.
- Tunes XGBoost hyperparameters with Optuna.
- Exports trained models for realtime monitoring.
- Scores Zeek connection logs with the same feature layout used during training.
- Optionally enriches realtime decisions with AbuseIPDB reputation data.

## Repository Layout

```text
DataPreprocessing/          Dataset loading, cleaning, label encoding, splits
ModelTraining/              Model training, evaluation, and export
HyperparameterOptimization/ Optuna-based XGBoost tuning
NetworkMonitor/             Realtime Zeek conn.log scoring
docs/                       Architecture and conference context
tests/                      Unit tests for core feature and evaluation behavior
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The historical Conda environment is still available in `environment.yml`, but the lightweight `requirements*.txt` files are the recommended starting point for local development and CI.

## Data

Place downloaded datasets under `DataPreprocessing/data/`. The original experiments used public IDS datasets such as CICIDS2017, UNSW-NB15, ToN-IoT, and related traffic captures. Large datasets and generated preprocessing outputs are intentionally ignored by Git.

```bash
python DataPreprocessing/main.py
```

This generates:

- `DataPreprocessing/preprocessed_data/X_train_stratify.csv`
- `DataPreprocessing/preprocessed_data/X_test_stratify.csv`
- `DataPreprocessing/preprocessed_data/y_train_encoded_stratify.csv`
- `DataPreprocessing/preprocessed_data/y_test_encoded_stratify.csv`
- `DataPreprocessing/model_mappings/mappings.json`

## Train

```bash
python -m ModelTraining.main
```

XGBoost uses CPU-friendly defaults. To use a CUDA-capable XGBoost install:

```bash
XGBOOST_DEVICE=cuda python -m ModelTraining.main
```

To enable RAPIDS cuML for Random Forest or SVM:

```bash
USE_CUML=true python -m ModelTraining.main
```

## Optimize XGBoost

```bash
python HyperparameterOptimization/optuna_gpu.py --n-trials 100 --device cpu
```

Use `--device cuda` only when XGBoost is installed with compatible GPU support.

## Run Realtime Monitoring

Install and run Zeek separately, then point the monitor at the active log directory:

```bash
python -m NetworkMonitor.main \
  --models-dir NetworkMonitor/models \
  --mappings-path DataPreprocessing/model_mappings/mappings.json \
  --log-dir /opt/zeek/logs/current
```

Optional AbuseIPDB enrichment can be enabled without committing secrets:

```bash
export ABUSEIPDB_API_KEY="..."
```

## Test

```bash
ruff check .
pytest
```

## Scope

This is the revitalized public UCC 2024 IDS repository. The broader AICCC autonomous-agent cyber-defense system is implemented as a separate public project that integrates this IDS component with routing, malware analysis, auditability, and human oversight.
