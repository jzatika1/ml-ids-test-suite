# Architecture

The repository is organized around a reproducible IDS workflow:

1. `DataPreprocessing/` normalizes supported public datasets into a compact feature set.
2. `ModelTraining/` trains and evaluates ML models, with XGBoost as the primary model.
3. `HyperparameterOptimization/` runs Optuna studies for XGBoost tuning.
4. `NetworkMonitor/` reads Zeek `conn.log`, extracts the same features used during training, scores traffic with the exported model, and optionally enriches IP reputation with AbuseIPDB.

## Feature Contract

Training and realtime inference use these columns in order:

- `Destination Port`
- `Source Port`
- `Protocol`
- `Total Fwd Bytes`
- `Total Bwd Bytes`
- `Flow Duration`

Protocol encoding is stable across preprocessing and realtime monitoring:

- `icmp` = `0`
- `tcp` = `1`
- `udp` = `2`

## Public Repo Boundaries

This repository remains focused on the UCC 2024 IDS work. The AICCC autonomous-agent system is implemented as a separate repository that imports or adapts this network IDS component alongside malware-analysis and orchestration modules.
