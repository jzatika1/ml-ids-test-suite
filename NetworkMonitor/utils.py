import ipaddress
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def to_numeric(value: Any, default=0):
    """
    Safely convert a string to an integer or float. If conversion fails, return a default value.

    Parameters:
    - value: str, the string value to convert.
    - default: int or float, the default value to return if conversion fails.

    Returns:
    - int or float, the converted number or the default value.
    """
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return value

    value = str(value).strip()
    if value in {"", "-", "(empty)"}:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
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
    path = Path(filepath)
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.warning("Mapping file not found: %s", path)
        return {}
    except json.JSONDecodeError:
        logger.warning("Error decoding JSON from %s", path)
        return {}


def save_mappings(data, filepath):
    """
    Save data to a JSON file.

    Parameters:
    - data: dict, the data to save.
    - filepath: str, the path to the JSON file where data will be saved.
    """
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except OSError as exc:
        logger.error("Error saving mapping data to %s: %s", path, exc)
