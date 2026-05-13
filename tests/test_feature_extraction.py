from sklearn.preprocessing import LabelEncoder

from NetworkMonitor.feature_extraction import FEATURE_COLUMNS, extract_features, features_to_frame
from NetworkMonitor.utils import to_numeric


def test_to_numeric_handles_zeek_empty_values():
    assert to_numeric("-") == 0
    assert to_numeric("(empty)") == 0
    assert to_numeric("42") == 42
    assert to_numeric("3.5") == 3.5


def test_extract_features_from_zeek_row():
    encoder = LabelEncoder()
    encoder.fit(["tcp", "udp", "icmp"])
    row = {
        "id.resp_p": "443",
        "id.orig_p": "51515",
        "proto": "tcp",
        "duration": "1.5",
        "orig_bytes": "100",
        "resp_bytes": "-",
    }

    features = extract_features(row, encoder)
    assert features == {
        "Destination Port": 443,
        "Source Port": 51515,
        "Protocol": 1,
        "Total Fwd Bytes": 100,
        "Total Bwd Bytes": 0,
        "Flow Duration": 1_500_000.0,
    }


def test_features_to_frame_uses_training_order():
    frame = features_to_frame({"Source Port": 1, "Destination Port": 2})

    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.loc[0, "Destination Port"] == 2
    assert frame.loc[0, "Total Fwd Bytes"] == 0
