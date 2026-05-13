import json
from pathlib import Path

DEFAULT_MAPPINGS_DIR = Path(__file__).resolve().parents[2] / "DataPreprocessing" / "model_mappings"


def num_classes(mappings_dir=DEFAULT_MAPPINGS_DIR):
    mappings_path = Path(mappings_dir)
    if mappings_path.is_file():
        json_path = mappings_path
    else:
        json_path = next(iter(sorted(mappings_path.glob("*.json"))), None)

    if json_path is None:
        raise FileNotFoundError(f"No mapping JSON file found in {mappings_path}.")

    with json_path.open("r", encoding="utf-8") as file:
        mappings = json.load(file)

    return len(mappings)
