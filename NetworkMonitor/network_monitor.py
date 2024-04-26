import asyncio
import os
import xgboost as xgb
import json

from models_loader import load_models
from feature_extraction import extract_features
from abuseipdb import get_abuseipdb_info
from utils import ip_to_int, load_mappings
from sklearn.preprocessing import LabelEncoder

# Data handling and analysis
import numpy as np
import pandas as pd

from io import StringIO

# Suppress specific warnings
import warnings
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)

class RealTimeEvaluator:
    def __init__(self, mappings, threshold=1, report_threshold=100):
        self.mappings = mappings
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        self.threshold = threshold  # AbuseIPDB score threshold for malicious IPs
        self.report_threshold = report_threshold  # Threshold for number of reports to consider the IP as malicious

    def update_metrics(self, predicted_class, actual_score, total_reports):
        """
        Update the evaluation metrics based on the single new prediction.
        :param predicted_class: The predicted class index (0 is benign, any other is malicious)
        :param actual_score: The AbuseIPDB reputation score
        :param total_reports: Total number of abuse reports for the IP
        """
        
        # This code will run in the Python environment where the file is located
        try:
            non_malicious_mapping = self.mappings['Non-Malicious']
            
            # Check if the prediction is non-malicious (10) or something else
            is_predicted_malicious = int(predicted_class != non_malicious_mapping)
        except FileNotFoundError:
            print(f"File not found: {mapping_file_path}")

        # Determine if actually malicious based on score or number of reports
        is_actual_malicious = int(actual_score >= self.threshold or total_reports > self.report_threshold)
        
        # Update metrics based on comparison
        if is_predicted_malicious == 1 and is_actual_malicious == 1:
            self.tp += 1
        elif is_predicted_malicious == 1 and is_actual_malicious == 0:
            self.fp += 1
        elif is_predicted_malicious == 0 and is_actual_malicious == 0:
            self.tn += 1
        elif is_predicted_malicious == 0 and is_actual_malicious == 1:
            self.fn += 1

    def calculate_metrics(self):
        """
        Calculate evaluation metrics based on current counts.
        :return: Dictionary with current evaluation metrics
        """
        try:
            precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
            recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn) if (self.tp + self.fp + self.tn + self.fn) > 0 else 0
        except ZeroDivisionError:
            precision, recall, f1_score, accuracy = 0, 0, 0, 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn
        }

class NetworkMonitor:
    def __init__(self, models_directory, mappings_path):
        self.models = load_models(models_directory)
        self.mappings = load_mappings(mappings_path)
        self.evaluator = RealTimeEvaluator(self.mappings)  # Instantiate the RealTimeEvaluator here
        
        # Invert the mappings to map from numeric codes to string labels
        self.inverse_mappings = {v: k for k, v in self.mappings.items()}
        self.last_read_position = 0
        self.last_known_size = 0
        self.local_ip_address = '10.0.0.125'  # Customize this to match your network configuration
        
        # Protocol label encoding
        self.protocol_encoder = LabelEncoder()
        self.protocol_encoder.fit(['tcp', 'udp', 'icmp'])

    async def monitor_logs(self, log_dir="/opt/zeek/logs/current", poll_interval=0.1):
        print(f"Monitoring {log_dir} for new log data...")
        
        while True:
            filepath = os.path.join(log_dir, 'conn.log')
            try:
                current_size = os.path.getsize(filepath)
                if current_size < self.last_known_size:
                    print("Detected conn.log rotation.")
                    self.last_read_position = 0
                self.last_known_size = current_size
                await self.process_log_data(filepath)
            except FileNotFoundError:
                print("conn.log file not found. It might be rotating or deleted.")
            except Exception as e:
                print(f"Error monitoring log files: {e}")
            await asyncio.sleep(poll_interval)

    async def process_log_data(self, filepath):
        """
        Process new log entries from the given file path, integrating feature extraction, model predictions,
        and outlier detection to assess network traffic.
        """
        with open(filepath, 'r') as file:
            file.seek(self.last_read_position)  # Move to the last read position
            lines = file.readlines()
            data_lines = [line for line in lines if not line.startswith('#')]
            log_data = ''.join(data_lines)
            self.last_read_position = file.tell()  # Update for the next read

        # Prepare the DataFrame with correct column names for analysis
        column_names = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents']
        log_df = pd.read_csv(StringIO(log_data), sep='\t', header=None, names=column_names, index_col=False)

        # Process each log entry
        for index, row in log_df.iterrows():
            if row['id.orig_h'] == self.local_ip_address:
                continue
    
            features = extract_features(row, self.protocol_encoder)
            features_df = pd.DataFrame([features])
            dtest = xgb.DMatrix(features_df)
            
            # This gets the probabilities as you've shown
            probabilities = self.models['xgboost'].predict(dtest)
            
            # To get the class with the maximum probability for each prediction
            predicted_classes = np.argmax(probabilities, axis=1)[0]
            
            ip_reputation_info = get_abuseipdb_info(row['id.orig_h'])
            abuse_score = ip_reputation_info.get('reputation', 0)
            total_reports = ip_reputation_info.get('totalReports', 0)
    
            self.evaluator.update_metrics(predicted_classes, abuse_score, total_reports)
    
            if index % 50 == 0:  # Adjusted for more frequent feedback during testing
                current_metrics = self.evaluator.calculate_metrics()
                print(current_metrics)
            
    async def run(self, log_dir="/opt/zeek/logs/current"):
        await self.monitor_logs(log_dir)