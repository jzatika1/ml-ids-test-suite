import asyncio
import logging
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

try:
    from .abuseipdb import get_abuseipdb_info
    from .feature_extraction import extract_features, features_to_frame
    from .models_loader import load_models
    from .utils import load_mappings, to_numeric
except ImportError:  # pragma: no cover - supports direct script execution
    from abuseipdb import get_abuseipdb_info
    from feature_extraction import extract_features, features_to_frame
    from models_loader import load_models
    from utils import load_mappings, to_numeric

logger = logging.getLogger(__name__)

ZEEK_CONN_COLUMNS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
]


class RealTimeEvaluator:
    def __init__(self, mappings, threshold=1, report_threshold=100):
        self.mappings = mappings
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.threshold = threshold
        self.report_threshold = report_threshold

    def _non_malicious_code(self):
        for label in ("Non-Malicious", "Non-malicious", "Benign", "BENIGN", "normal"):
            if label in self.mappings:
                return self.mappings[label]
        return 0

    def update_metrics(self, predicted_class, actual_score, total_reports):
        """
        Update metrics for a single realtime prediction.

        AbuseIPDB is used as a weak external signal for approximate realtime
        feedback, not as authoritative ground truth.
        """
        non_malicious_mapping = self._non_malicious_code()
        is_predicted_malicious = int(predicted_class != non_malicious_mapping)
        abuse_score = to_numeric(actual_score, 0)
        report_count = to_numeric(total_reports, 0)
        is_actual_malicious = int(
            abuse_score >= self.threshold or report_count > self.report_threshold
        )

        if is_predicted_malicious == 1 and is_actual_malicious == 1:
            self.tp += 1
        elif is_predicted_malicious == 1 and is_actual_malicious == 0:
            self.fp += 1
        elif is_predicted_malicious == 0 and is_actual_malicious == 0:
            self.tn += 1
        elif is_predicted_malicious == 0 and is_actual_malicious == 1:
            self.fn += 1

    def calculate_metrics(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )
        denominator = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / denominator if denominator > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


class NetworkMonitor:
    def __init__(self, models_directory, mappings_path, local_ip_address=None):
        self.models = load_models(models_directory)
        self.mappings = load_mappings(mappings_path)
        self.evaluator = RealTimeEvaluator(self.mappings)
        self.inverse_mappings = {v: k for k, v in self.mappings.items()}
        self.last_read_position = 0
        self.last_known_size = 0
        self.local_ip_address = local_ip_address or os.getenv("LOCAL_IP_ADDRESS", "")

        self.protocol_encoder = LabelEncoder()
        self.protocol_encoder.fit(["tcp", "udp", "icmp"])

    async def monitor_logs(self, log_dir="/opt/zeek/logs/current", poll_interval=0.1):
        logger.info("Monitoring %s for new Zeek conn.log data", log_dir)

        while True:
            filepath = Path(log_dir) / "conn.log"
            try:
                current_size = filepath.stat().st_size
                if current_size < self.last_known_size:
                    logger.info("Detected conn.log rotation.")
                    self.last_read_position = 0
                self.last_known_size = current_size
                await self.process_log_data(filepath)
            except FileNotFoundError:
                logger.warning("conn.log was not found; it may be rotating or not created yet.")
            except Exception as exc:
                logger.exception("Error monitoring Zeek logs: %s", exc)
            await asyncio.sleep(poll_interval)

    async def process_log_data(self, filepath):
        """
        Process newly appended Zeek conn.log entries.
        """
        path = Path(filepath)
        with path.open("r", encoding="utf-8", errors="replace") as file:
            file.seek(self.last_read_position)
            lines = file.readlines()
            self.last_read_position = file.tell()

        data_lines = [line for line in lines if line.strip() and not line.startswith("#")]
        if not data_lines:
            return []

        log_data = "".join(data_lines)
        log_df = pd.read_csv(
            StringIO(log_data),
            sep="\t",
            header=None,
            names=ZEEK_CONN_COLUMNS,
            index_col=False,
        )

        return [self.process_log_row(row) for _, row in log_df.iterrows()]

    def process_log_row(self, row):
        if self.local_ip_address and row["id.orig_h"] == self.local_ip_address:
            return None

        if "xgboost" not in self.models:
            raise RuntimeError(
                "No xgboost model was loaded; train/export a model before monitoring."
            )

        features = extract_features(row, self.protocol_encoder)
        features_df = features_to_frame(features)
        dtest = xgb.DMatrix(features_df)
        probabilities = np.asarray(self.models["xgboost"].predict(dtest))
        if probabilities.ndim == 1:
            predicted_class = (
                int(probabilities[0] >= 0.5)
                if probabilities.size == 1
                else int(np.argmax(probabilities))
            )
        else:
            predicted_class = int(np.argmax(probabilities, axis=1)[0])

        ip_reputation_info = get_abuseipdb_info(row["id.orig_h"]) or {}
        abuse_score = ip_reputation_info.get("reputation", 0)
        total_reports = ip_reputation_info.get("totalReports", 0)

        self.evaluator.update_metrics(predicted_class, abuse_score, total_reports)
        metrics = self.evaluator.calculate_metrics()
        logger.info(
            "Prediction src=%s class=%s label=%s metrics=%s",
            row["id.orig_h"],
            predicted_class,
            self.inverse_mappings.get(predicted_class, "unknown"),
            metrics,
        )

        return {
            "source_ip": row["id.orig_h"],
            "predicted_class": predicted_class,
            "predicted_label": self.inverse_mappings.get(predicted_class, "unknown"),
            "features": features,
            "metrics": metrics,
        }

    async def run(self, log_dir="/opt/zeek/logs/current", poll_interval=0.1):
        await self.monitor_logs(log_dir, poll_interval=poll_interval)
