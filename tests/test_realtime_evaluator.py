from NetworkMonitor.network_monitor import RealTimeEvaluator


def test_realtime_evaluator_tracks_confusion_counts():
    evaluator = RealTimeEvaluator({"Non-Malicious": 0, "Scanning": 1})

    evaluator.update_metrics(predicted_class=1, actual_score=95, total_reports=5)
    evaluator.update_metrics(predicted_class=0, actual_score=0, total_reports=0)

    assert evaluator.calculate_metrics() == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 0,
    }
