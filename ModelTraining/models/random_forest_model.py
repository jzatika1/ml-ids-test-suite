from cuml.ensemble import RandomForestClassifier
import cudf

def train_random_forest(X_train, y_train):
    # Convert data to cuDF DataFrame for GPU acceleration
    X_train_gdf = cudf.DataFrame(X_train).astype('float32')
    y_train_gdf = cudf.Series(y_train).astype('float32')
    
    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=23, random_state=42, n_streams=1)
    rf_classifier.fit(X_train_gdf, y_train_gdf)
    
    return rf_classifier