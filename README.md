# Disclaimer: 
We are currently working on providing the datasets used in the experiments on a hosting platform. Until this is completed, the data required for model training will not be accessible. Thank you for your patience.
In the meantime, you can download the CICIDS'17 dataset and extract the CSV files into the data folder within DataPreprocessing.

# Network Monitor with Anomaly Detection

This repository hosts the code for a sophisticated network monitoring and anomaly detection system designed to identify malicious or anomalous network traffic using machine learning models. The system utilizes a modular architecture for ease of maintenance and scalability.

## Features

- Real-time monitoring of network logs.
- Feature extraction from network traffic data.
- Anomaly detection using machine learning models including Random Forests, Support Vector Machines, Neural Networks, and XGBoost.
- Scalable architecture supporting asynchronous processing of network logs.

## Getting Started

Before running any of the files, ensure you set up the environment and models as described. Follow the steps below:

### Prerequisites

Before installing the project dependencies, you need to have Conda installed. If you do not have Conda installed, you can install Miniconda by following these steps:

1. **Download Miniconda Installer:**
   Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.

   - **Windows:** Download the `.exe` file and open it to start the installer.
   - **macOS:** Download the `.pkg` file for a graphical installer or the `.sh` file for a command line installation.
   - **Linux:** Download the `.sh` file.

2. **Install Miniconda:**
   Open a terminal (or command prompt on Windows) and navigate to the directory where the downloaded file is located.

   - **For macOS and Linux:**
     ```shell
     bash Miniconda3-latest-Linux-x86_64.sh  # Adjust the filename as needed
     ```
     Follow the on-screen instructions. It is recommended to allow the installer to initialize Miniconda by running `conda init`.

   - **For Windows:**
     Double-click the downloaded `.exe` file and follow the on-screen instructions.

3. **Verify Installation:**
   Restart your terminal (or command prompt) and type the following command to see if Conda was installed correctly:
   ```shell
   conda --version
   ```

### Installation

1. Clone the repository to your local machine:

    ```shell
    git clone https://github.com/jzatika1/ml-ids-test-suite.git
    cd ml-ids-test-suite
    ```

2. Install the required environment:

    ```shell
    conda env create -f environment.yml
    conda activate ml-ids-env
    ```

### Model Training

Before starting the network monitor, it is crucial to train the machine learning models.

1. **Prepare the Data:**

    Before training the models, you need to preprocess the data to ensure it's in the right format for training:

    ```shell
	cd DataPreprocessing/
    python main.py
    ```

    This script will combine and preprocess the necessary datasets using multiple processes to speed up the preparation.

2. **Navigate to the Training Program Directory:**

    ```shell
	cd ModelTraining/
    ```

3. **Train the Default XGBoost Model:**

    ```shell
    python main.py
    ```

    If you wish to train other models like SVM, Random Forests, or Neural Networks:

    - Edit the `config.ini` file within the `training_program/config` directory.
    - Set the appropriate model to `True` under the `[models]` section to enable training.

### Running the Monitor

To start monitoring, navigate back to the root directory and run the following command:

```shell
cd NetworkMonitor
sudo python main.py
```