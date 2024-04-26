import configparser
import os

# Import specific training functions or classes from models
from models import xgboost_model, random_forest_model, neural_network_model, svm_model
from evaluate_model.calculate_results import calculate_results
from data.data_processor import load_data
from data.export_model import export_model

def main():
    config_path = 'config/config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Dictionary of model functions for easy access
    model_functions = {
        'xgboost': xgboost_model.train_xgboost,
        'random_forest': random_forest_model.train_random_forest,
        'neural_network': neural_network_model.train_neural_network,
        'svm': svm_model.train_svm
    }

    results = {}

    # Loop through each model configuration
    for model_name, train_func in model_functions.items():
        if config.getboolean('models', model_name):
            print(f"Training {model_name}...")
            trained_model = train_func(X_train, y_train)
            metrics = calculate_results(trained_model, X_test, y_test)
            results[model_name] = metrics
            
            print(f"Evaluation Metrics for {model_name.capitalize()} Model:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    formatted_value = f"{value * 100:.3f}%" if metric != "ROC-AUC Score" else f"{value:.3f}"
                else:
                    formatted_value = value  # Keep the string as is if it's not a float (like 'Not computed')
                print(f"  {metric}: {formatted_value}")
            
            # Export the model after evaluation
            export_model(trained_model, model_name)

if __name__ == "__main__":
    main()