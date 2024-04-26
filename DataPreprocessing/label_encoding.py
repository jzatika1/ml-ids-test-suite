import json
import time
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode_labels(y_train, y_test, save_folder='model_mappings', save_file='mappings.json'):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    # Retrieve the mapping of encoded labels
    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    print("Label encoding mapping:", mapping)
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Construct the file path using os.path.join()
    save_path = os.path.join(save_folder, save_file)
    
    # Save the mappings to a JSON file
    with open(save_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    
    return y_train_encoded, y_test_encoded