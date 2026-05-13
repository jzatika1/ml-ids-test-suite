import pandas as pd

try:
    from .utils import to_numeric
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import to_numeric

FEATURE_COLUMNS = [
    "Destination Port",
    "Source Port",
    "Protocol",
    "Total Fwd Bytes",
    "Total Bwd Bytes",
    "Flow Duration",
]

PROTOCOL_FALLBACKS = {
    "icmp": 0,
    "tcp": 1,
    "udp": 2,
}


def encode_protocol(protocol, protocol_encoder):
    """Encode a Zeek protocol value using a fitted encoder, falling back safely."""
    protocol = str(protocol or "").strip().lower()
    if not protocol:
        return -1

    try:
        return int(protocol_encoder.transform([protocol])[0])
    except ValueError:
        return PROTOCOL_FALLBACKS.get(protocol, -1)


def extract_features(log_row, protocol_encoder):
    """
    Extract features from a single row of log data.

    Parameters:
    - log_row: dict, a dictionary representing a single row of log data.
    - protocol_encoder: LabelEncoder, an instance of LabelEncoder pre-fitted with protocol types.

    Returns:
    - features: dict, a dictionary of extracted features ready for model input.
    """
    dst_port = to_numeric(log_row.get("id.resp_p"), 0)
    src_port = to_numeric(log_row.get("id.orig_p"), 0)
    protocol = encode_protocol(log_row.get("proto"), protocol_encoder)
    duration_in_seconds = to_numeric(log_row.get("duration"), 0.0)
    duration = duration_in_seconds * 1_000_000

    total_fwd_bytes = to_numeric(log_row.get("orig_bytes"), 0)
    total_bwd_bytes = to_numeric(log_row.get("resp_bytes"), 0)

    features = {
        "Destination Port": dst_port,
        "Source Port": src_port,
        "Protocol": protocol,
        "Total Fwd Bytes": total_fwd_bytes,
        "Total Bwd Bytes": total_bwd_bytes,
        "Flow Duration": duration,
    }

    return features


def features_to_frame(features):
    """Create a single-row DataFrame in the expected training-column order."""
    return pd.DataFrame([{column: features.get(column, 0) for column in FEATURE_COLUMNS}])
