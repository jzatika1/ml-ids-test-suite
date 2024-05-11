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

### Installing Zeek

1. Ensure your system is supported (Linux, macOS).
2. Install necessary dependencies through your package manager.
3. Download Zeek from [Zeek.org](https://zeek.org/get-zeek/) or install via package manager.
4. Follow the detailed installation instructions provided on the Zeek website or through the installation package.

### Configuring Zeek as a System Service

1. To configure Zeek as a service, create a new service file at `/etc/systemd/system/zeek.service` with the following content:

	```ini
	[Unit]
	Description=Zeek Network Intrusion Detection System (NIDS)
	After=network.target

	[Service]
	Type=forking
	User=root
	Group=zeek
	Environment=HOME=/nsm/zeek/spool
	ExecStart=/opt/zeek/bin/zeekctl deploy
	ExecStop=/opt/zeek/bin/zeekctl stop

	[Install]
	WantedBy=multi-user.target
	```

2. Enable and start the service:
	```bash
	sudo systemctl enable zeek
	sudo systemctl start zeek
	```

### Configure Network Settings in Zeek

1. Modify the `node.cfg` file to specify your network interface:

	```bash
	sudo nano /opt/zeek/etc/node.cfg
	```

2. Change the `interface` setting to match your network interface (e.g., `eth0`).

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

### Downloading Datasets

To run this program, you'll need to download several datasets from Kaggle. Below are the steps to download these datasets:

1. If you don't already have a Kaggle account, you will need to create one. Go to [Kaggle](https://www.kaggle.com) and sign up.

2. Once you have a Kaggle account, use the following links to access each dataset. Click the "Download" button on the dataset page to download the zip files:

- **RouteSmart Dataset:** [Download Link](https://www.kaggle.com/datasets/janthonyzatika/routesmart)
- **CICIDS2017 Dataset:** [Download Link](https://www.kaggle.com/datasets/cicdataset/cicids2017/)
- **ToN_IoT Train-Test Network Dataset:** [Download Link](https://www.kaggle.com/datasets/fadiabuzwayed/ton-iot-train-test-network)
- **UNSW-NB15 Dataset:** [Download Link](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)

3. After downloading, unzip each dataset. Move the CSV files (or other relevant files, depending on the dataset structure) into the following folder within your local repository:

	```shell
	cd ml-ids-test-suite1/DataPreprocessing/data/
	```

4. Make sure that all dataset files are correctly placed in the `data` folder. The program will expect to find them there.

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
