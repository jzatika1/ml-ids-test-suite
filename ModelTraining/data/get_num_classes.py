import os
import json

def num_classes():
    # Relative path to your mappings directory
    mappings_dir_path = '../DataPreprocessing/model_mappings/'

    # Get the absolute path to the mappings directory
    absolute_mappings_dir_path = os.path.abspath(mappings_dir_path)

    # Find the first JSON file in the directory
    json_file = next((file for file in os.listdir(absolute_mappings_dir_path) if file.endswith('.json')), None)

    if json_file:
        # Load mappings file
        with open(os.path.join(absolute_mappings_dir_path, json_file), 'r') as f:
            mappings = json.load(f)

        # Determine the number of unique classes from the mappings
        num_classes = len(mappings)
        return num_classes
    else:
        print("No JSON file found in the directory.")
        return None