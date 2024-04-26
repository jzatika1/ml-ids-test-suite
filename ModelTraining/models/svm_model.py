from cuml.svm import SVC
from sklearn.model_selection import train_test_split
import cudf

def train_svm(X_train, y_train):
    """
    Trains an SVM model using cuML on GPU-accelerated data structures.

    Args:
    X_train (np.array or pd.DataFrame): Training data features.
    y_train_encoded (np.array): Encoded training data labels.

    Returns:
    svm (cuml.svm.SVC): Trained SVM model.
    """

    # Stratified Sampling to maintain precise proportions in a smaller subset
    X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, 
                                                            test_size=0.30, stratify=y_train, 
                                                            random_state=42)
    
    # Convert sampled data to cuDF DataFrames for GPU acceleration
    X_train_sample_gdf = cudf.DataFrame(X_train_sample)
    y_train_sample_gdf = cudf.Series(y_train_sample)
    
    # Create and train the SVM with RBF kernel and probability estimates
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_sample_gdf, y_train_sample_gdf)

    return svm

