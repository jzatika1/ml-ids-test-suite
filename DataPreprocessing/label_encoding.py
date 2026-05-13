import json
from pathlib import Path

from sklearn.preprocessing import LabelEncoder


def encode_labels(y_train, y_test, save_folder="model_mappings", save_file="mappings.json"):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_)), strict=True))
    print("Label encoding mapping:", mapping)

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / save_file

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)

    return y_train_encoded, y_test_encoded
