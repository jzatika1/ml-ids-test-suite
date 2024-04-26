import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder

def load_file(filepath):
    try:
        # Read the first line to make a guess about the presence of a header
        with open(filepath, 'r', encoding="utf-8-sig") as file:
            first_line = file.readline().strip()
            # Check if the majority of the first line's elements contain digits
            elements = first_line.split(',')
            has_header = sum(element.strip()[0].isdigit() for element in elements if element.strip()) < len(elements) / 2

        desired_columns = ['Destination Port', 'Source Port', 'Protocol', 'Total Fwd Bytes', 'Total Bwd Bytes', 'Flow Duration', 'Label']

        if has_header:
            # If there is a header, load only the first few rows to determine the columns present
            try:
                # First attempt with the primary desired columns
                desired_columns_primary = [' Destination Port', ' Source Port', ' Protocol', ' Subflow Fwd Bytes', ' Subflow Bwd Bytes', ' Flow Duration', ' Label']
                data = pd.read_csv(filepath, usecols=desired_columns_primary, low_memory=False, encoding="ISO-8859-1")
                
                # Ensure columns are properly named
                data.columns = data.columns.str.strip()
                
                data.rename(columns={
                    'Subflow Fwd Bytes': 'Total Fwd Bytes',
                    'Subflow Bwd Bytes': 'Total Bwd Bytes'
                }, inplace=True)
                
                # Remove all rows with NaN values
                data = data.dropna()
                
                # Remove rows where both 'Source Port' and 'Destination Port' are zero
                data = data[(data['Source Port'] != 0) | (data['Destination Port'] != 0)]
                
                # Convert 'Source Port', 'Destination Port', and 'Protocol' columns to integer type
                data['Source Port'] = data['Source Port'].astype(int)
                data['Destination Port'] = data['Destination Port'].astype(int)
                data['Protocol'] = data['Protocol'].astype(int)
            except ValueError as e:
                try:
                    # Fallback to a secondary set of desired columns if there's an error
                    desired_columns_secondary = ['dst_port', 'src_port', 'proto', 'dst_bytes', 'src_bytes', 'duration', 'type']
                    data = pd.read_csv(filepath, usecols=desired_columns_secondary, low_memory=False, encoding="ISO-8859-1")
                    
                    # Rename columns to match the desired format
                    rename_mapping = { 
                        'dst_port': 'Destination Port',
                        'src_port': 'Source Port',
                        'proto': 'Protocol', 
                        'dst_bytes': 'Total Fwd Bytes', 
                        'src_bytes': 'Total Bwd Bytes',
                        'duration': 'Flow Duration', 
                        'type': 'Label'
                    }
                    data.rename(columns=rename_mapping, inplace=True)
                except ValueError as e:
                    # Fallback to a third set of desired columns if there's an error
                    desired_columns_third = ['target_port', 'client_port', 'request_proto', 'request_processing_time', 'target_processing_time', 'response_processing_time', 'received_bytes', 'sent_bytes']
                    data = pd.read_csv(filepath, usecols=desired_columns_third, low_memory=False, encoding="ISO-8859-1")
                    
                    # Remove rows where 'target_port' is missing
                    data = data.dropna(subset=['target_port'])
                    
                    # Renaming the columns
                    data.rename(columns={
                        'received_bytes': 'src_bytes',
                        'sent_bytes': 'dst_bytes'
                    }, inplace=True)
                    
                    # Replacing all values in 'request_proto' with 'tcp'
                    data['request_proto'] = 'tcp'
                    
                    # Adding the columns to get 'duration'
                    data['duration'] = data['request_processing_time'] + data['target_processing_time'] + data['response_processing_time']
                    
                    # Convert seconds to microseconds by multiplying by 1,000,000 (1 second = 1,000,000 microseconds)
                    data['duration'] = data['duration'] * 1_000_000
                    
                    # Dropping the original columns
                    data.drop(columns=['request_processing_time', 'target_processing_time', 'response_processing_time'], inplace=True)
                    
                    # Adding a new column 'Label' with all values set to 'malicious'
                    data['Label'] = 1
                    
                    # Rename columns to match the desired format
                    rename_mapping = { 
                        'target_port': 'Destination Port',
                        'client_port': 'Source Port',
                        'request_proto': 'Protocol',
                        'dst_bytes': 'Total Fwd Bytes', 
                        'src_bytes': 'Total Bwd Bytes', 
                        'duration': 'Flow Duration', 
                        'type': 'Label'
                    }
                    data.rename(columns=rename_mapping, inplace=True)
                    
                    # Convert 'Source Port', 'Destination Port', and 'Protocol' columns to integer type
                    data['Source Port'] = data['Source Port'].astype(int)
                    data['Destination Port'] = data['Destination Port'].astype(int)       
                
        else:
            # Load the dataset, assuming no headers
            data = pd.read_csv(filepath, header=None, low_memory=False, encoding="ISO-8859-1")
            
            # Define your desired column names
            desired_columns = ['Destination Port', 'Source Port', 'Protocol', 'Total Fwd Bytes', 'Total Bwd Bytes', 'Flow Duration', 'Label']
            
            # Define the original positions of these columns (0-based indexing)
            column_positions = [
                3,
                1,
                4,
                9,
                10,
                6,
                48,
            ]

            # Generate a list of new column names based on the existing number of columns in 'data'
            new_column_names = [f"Column_{i+1}" for i in range(len(data.columns))]
            
            # Assign the desired names to the specified positions
            for pos, name in zip(column_positions, desired_columns):
                new_column_names[pos] = name
            
            # Update the DataFrame's column names
            data.columns = new_column_names
            
            # Select only the columns specified in your position list
            # We use the 'desired_columns' directly since they are already aligned with 'column_positions'
            data = data[desired_columns]
            
            # Convert seconds to microseconds by multiplying by 1,000,000 (1 second = 1,000,000 microseconds)
            data['Flow Duration'] = data['Flow Duration'] * 1_000_000
            
        # Ensure columns are properly named
        data.columns = data.columns.str.strip()
        
        # Create a copy of the feature data to avoid SettingWithCopyWarning
        X = data[desired_columns].copy()
        
        # Initialize an empty DataFrame for labels
        y = pd.DataFrame()
        
        # Check for 'Label' column and use it
        if 'Label' in data.columns:
            y = data['Label'].copy()
            X.drop('Label', axis=1, inplace=True)  # Remove the 'Label' column from X

            # Define the protocols we care about
            protocols_we_care_about = ['tcp', 'udp', 'icmp', 6, 17, 1]
            
            # Create a boolean mask for rows with the desired protocols
            mask = X['Protocol'].isin(protocols_we_care_about)
            
            # Apply the mask to both X and y to keep only the rows with the desired protocols
            X = X[mask]
            y = y[mask]
        
        y = y.replace('normal', 'Non-malicious')
        y = y.replace('BENIGN', 'Non-malicious')
        y = y.replace({0: 'Non-malicious', 1: 'Malicious (Unlabeled)'})

        # Convert ports to numeric, handling any non-numeric gracefully
        X['Source Port'] = pd.to_numeric(X['Source Port'], errors='coerce')
        X['Destination Port'] = pd.to_numeric(X['Destination Port'], errors='coerce')
        
        # Encode 'Protocol' as categorical
        protocol_encoder = LabelEncoder()
        X['Protocol'] = protocol_encoder.fit_transform(X['Protocol'])
        
        # Convert 'Flow Duration'
        X['Flow Duration'] = pd.to_numeric(X['Flow Duration'], errors='coerce')
        
        # Convert packets to numeric, handling any non-numeric gracefully
        X['Total Fwd Bytes'] = pd.to_numeric(X['Total Fwd Bytes'], errors='coerce')
        X['Total Bwd Bytes'] = pd.to_numeric(X['Total Bwd Bytes'], errors='coerce')
        
        # Find valid indices where neither X nor y has NaN values
        # This combines checking for non-NaN rows in both X and y
        valid_indices = ~X.isnull().any(axis=1) & y.notna()
        
        # Apply the valid indices to filter both X and y
        X_clean = X[valid_indices].copy()
        y_clean = y[valid_indices].copy()
        
        return X_clean, y_clean

    except Exception as e:
        print(f"Failed to load {filepath} due to: {e}")
        return pd.DataFrame(), pd.Series()

def concatenate_results(results):
    """Concatenate all X and y results from multiprocessing."""
    X_frames = [result[0] for result in results if not result[0].empty]
    y_frames = [result[1] for result in results if not result[1].empty]

    # Concatenate all X DataFrames and all y DataFrames separately
    X_combined = pd.concat(X_frames, ignore_index=True) if X_frames else pd.DataFrame()
    y_combined = pd.concat(y_frames, ignore_index=True) if y_frames else pd.Series(dtype='object')

    return X_combined, y_combined

def load_files(filepaths):
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(load_file, filepaths)
    
    X_combined, y_combined = concatenate_results(results)
    return X_combined, y_combined