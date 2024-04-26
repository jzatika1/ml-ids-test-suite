import pandas as pd
from utils import to_numeric  # Assuming you have a utility function for conversion

def extract_features(log_row, protocol_encoder):
    """
    Extract features from a single row of log data.

    Parameters:
    - log_row: dict, a dictionary representing a single row of log data.
    - protocol_encoder: LabelEncoder, an instance of LabelEncoder pre-fitted with protocol types.

    Returns:
    - features: dict, a dictionary of extracted features ready for model input.
    """
    # NOTE I want to make sure that if I do not have a specific feature in my dataset that I return an error stating that the feature was not present.
    # or pass so that it does not hang my program...
    
    dst_port = to_numeric(log_row.get('id.resp_p'), 0)
    src_port = to_numeric(log_row.get('id.orig_p'), 0)
    protocol = protocol_encoder.transform([log_row.get('proto')])[0]  # Transform protocol to numeric
    duration_in_seconds = to_numeric(log_row.get('duration'), 0.0)
    duration = duration_in_seconds * 1_000_000 # This is for micro-seconds
    
    print(duration)
    
    #total_fwd_packets = to_numeric(log_row.get('orig_pkts'), 0)
    #total_bwd_packets = to_numeric(log_row.get('resp_pkts'), 0)
    
    total_fwd_bytes = to_numeric(log_row.get('orig_bytes'), 0)
    total_bwd_bytes = to_numeric(log_row.get('resp_bytes'), 0)

    # Create a dictionary for the extracted features
    features = {
        'Destination Port': dst_port,
        'Source Port': src_port,
        'Protocol': protocol,
        'Total Fwd Bytes': total_fwd_bytes,
        'Total Bwd Bytes': total_bwd_bytes,
        'Flow Duration': duration,
    }

    return features