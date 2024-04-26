import json
import ipaddress

def to_numeric(value, default=0):
    """
    Safely convert a string to an integer or float. If conversion fails, return a default value.

    Parameters:
    - value: str, the string value to convert.
    - default: int or float, the default value to return if conversion fails.

    Returns:
    - int or float, the converted number or the default value.
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return default

def ip_to_int(ip_str):
    """
    Convert an IP address from string format to its integer representation.

    Parameters:
    - ip_str: str, the IP address in string format.

    Returns:
    - int, the integer representation of the IP address.
    """
    try:
        return int(ipaddress.ip_address(ip_str))
    except ValueError:
        return 0

def load_mappings(filepath):
    """
    Load JSON-encoded data from a file.

    Parameters:
    - filepath: str, the path to the JSON file.

    Returns:
    - dict, the data loaded from the JSON file.
    """
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}")
        return {}

def save_mappings(data, filepath):
    """
    Save data to a JSON file.

    Parameters:
    - data: dict, the data to save.
    - filepath: str, the path to the JSON file where data will be saved.
    """
    try:
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")