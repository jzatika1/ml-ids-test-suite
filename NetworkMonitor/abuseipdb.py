import configparser
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlencode

import urllib3
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.ini"


def _neutral_reputation():
    return {"reputation": 0, "totalReports": 0}


def _load_api_key(config_path=CONFIG_PATH):
    env_key = os.getenv("ABUSEIPDB_API_KEY", "").strip()
    if env_key:
        return env_key

    config = configparser.ConfigParser()
    config.read(config_path)
    api_key = config.get("DEFAULT", "ABUSE_IPDB_API_KEY", fallback="").strip()
    if api_key == "API_KEY_HERE":
        return ""
    return api_key


def _safe_get(data, key, default):
    value = data.get(key)
    if value is None:
        return default
    return value


def get_abuseipdb_info(ip_address, api_key=None):
    api_key = api_key or _load_api_key()
    if not api_key:
        logger.debug("AbuseIPDB API key is not configured; returning neutral reputation.")
        return _neutral_reputation()

    field_string = urlencode({"ipAddress": ip_address, "maxAgeInDays": "180"})
    full_url = f"https://api.abuseipdb.com/api/v2/check?{field_string}"
    headers = {
        "Accept": "application/json",
        "Key": api_key,
    }

    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    timeout = Timeout(connect=3.0, read=2.0)
    http = urllib3.PoolManager(retries=retries, timeout=timeout)

    try:
        response = http.request("GET", full_url, headers=headers)
        if response.status != 200:
            logger.warning("AbuseIPDB returned HTTP %s.", response.status)
            return _neutral_reputation()

        payload = json.loads(response.data.decode("utf-8"))
        data = payload.get("data", {})
        return {
            "reputation": _safe_get(data, "abuseConfidenceScore", 0),
            "totalReports": _safe_get(data, "totalReports", 0),
        }
    except Exception as exc:
        logger.warning("Error fetching AbuseIPDB info for %s: %s", ip_address, exc)
        return _neutral_reputation()
