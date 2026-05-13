import os

try:
    import cudf
    from cuml.svm import SVC as GpuSVC
except ImportError:  # pragma: no cover - optional GPU dependency
    cudf = None
    GpuSVC = None

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def train_svm(X_train, y_train, sample_test_size=0.30, random_state=42):
    """
    Train an SVM on a stratified sample of the training set.

    Set USE_CUML=true to train with RAPIDS cuML when that stack is installed.
    """
    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        test_size=sample_test_size,
        stratify=y_train,
        random_state=random_state,
    )

    use_gpu = os.getenv("USE_CUML", "").lower() in {"1", "true", "yes"}
    if use_gpu and GpuSVC is None:
        raise ImportError("USE_CUML is enabled, but RAPIDS cuML/cudf is not installed.")

    if use_gpu:
        X_sample = cudf.DataFrame(X_sample)
        y_sample = cudf.Series(y_sample)
        model = GpuSVC(kernel="rbf", probability=True)
    else:
        model = SVC(kernel="rbf", probability=True)

    model.fit(X_sample, y_sample)
    return model
