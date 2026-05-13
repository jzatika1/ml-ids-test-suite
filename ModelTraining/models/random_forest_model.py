import os

try:
    import cudf
    from cuml.ensemble import RandomForestClassifier as GpuRandomForestClassifier
except ImportError:  # pragma: no cover - optional GPU dependency
    cudf = None
    GpuRandomForestClassifier = None

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=23, random_state=42):
    use_gpu = os.getenv("USE_CUML", "").lower() in {"1", "true", "yes"}
    if use_gpu and GpuRandomForestClassifier is None:
        raise ImportError("USE_CUML is enabled, but RAPIDS cuML/cudf is not installed.")

    if use_gpu:
        X_train = cudf.DataFrame(X_train).astype("float32")
        y_train = cudf.Series(y_train).astype("float32")
        model = GpuRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_streams=1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    return model
