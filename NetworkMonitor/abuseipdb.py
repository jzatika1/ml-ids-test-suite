import logging
import configparser
import json
import urllib3
import os

from urllib.parse import urlencode
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout

log_directory = 'logs/'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)  # Creates the log directory if it doesn't exist

# Configure logging to write to a file in the specified directory
log_file_path = os.path.join(log_directory, 'abuseipdb_info.log')
logging.basicConfig(
    filename=log_file_path,
    filemode='a',  # Append mode, which will add to the file (not overwrite)
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration file
config = configparser.ConfigParser()
config.read("config/config.ini")

def _safe_get(data, key, default):
    """Helper function to get the value from the data dictionary,
    returns default if the key is not found or the value is None."""
    value = data.get(key)
    if value is None:
        return default
    return value

def get_abuseipdb_info(ip_address):
    url = 'https://api.abuseipdb.com/api/v2/check'
    headers = {
        'Accept': 'application/json',
        'Key': config['DEFAULT']['ABUSE_IPDB_API_KEY']
    }
    field_string = urlencode({'ipAddress': ip_address, 'maxAgeInDays': '180'})
    full_url = f"{url}?{field_string}"

    # Configure retries and timeout
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    timeout = Timeout(connect=3.0, read=2.0)
   
    # Create a connection pool manager instance
    http = urllib3.PoolManager(retries=retries, timeout=timeout)

    try:
        response = http.request('GET', full_url, headers=headers)
        if response.status == 200:
            data = json.loads(response.data.decode('utf-8'))
            logger.info(data)  # Log the entire data
            
            # Extract the necessary information and return it
            return {
                'reputation': _safe_get(data.get('data', {}), 'abuseConfidenceScore', 'No reputation score available'),
                'totalReports': _safe_get(data.get('data', {}), 'totalReports', 'No total reports available'),
            }
        else:
            logger.error(f"Non-successful response status: {response.status}")
            return None
    except Exception as e:
        logger.exception(f"Error fetching abuseIPDB info: {e}")
        return {'reputation': 'API call failed', 'totalReports': 'API call failed'}