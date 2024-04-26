import time
from sklearn.model_selection import train_test_split

def split_data(X_combined, y_combined, test_size=0.20, random_state=42):
    # Right before splitting the dataset into training and testing sets:
    if y_combined.isnull().any().any():
        raise ValueError("Labels contain NaN values after preprocessing.")
    
    if X_combined.isnull().any().any():
        raise ValueError("Features contain NaN values after preprocessing.")
    
    # Split the dataset into training and testing sets with stratified sampling
    split_start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, stratify=y_combined, test_size=test_size, random_state=random_state)
    split_end_time = time.time()
  
    split_duration = split_end_time - split_start_time
    print(f"train_test_split took: {split_duration}")
    
    # Displaying the class distribution in the training set
    train_distribution = y_train.value_counts()
    print("\nTraining Set Class Distribution:\n", train_distribution)

    # Displaying the class distribution in the testing set
    test_distribution = y_test.value_counts()
    print("\nTesting Set Class Distribution:\n", test_distribution)

    # Optionally, show the proportion of classes in the splits to verify stratification
    train_proportion = y_train.value_counts(normalize=True)
    print("\nTraining Set Class Proportions:\n", train_proportion)
    test_proportion = y_test.value_counts(normalize=True)
    print("\nTesting Set Class Proportions:\n", test_proportion)
    
    return X_train, X_test, y_train, y_test