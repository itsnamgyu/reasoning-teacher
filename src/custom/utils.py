import json
import os
import warnings
from typing import Optional

from paths import get_finetune_data_path


def save_finetune_data(data, platform_key: str, finetune_key: str, strict=True) -> str:
    path = get_finetune_data_path(platform_key, finetune_key)
    print("Saving finetune data")
    print("-" * 80)
    print("Path:       {}".format(path))
    print("Samples:    {}".format(len(data["input"])))
    print("-" * 80)

    if os.path.exists(path):
        with open(path) as f:
            data_string = json.dumps(data, indent=4)
            existing_data_string = f.read()
            if data_string != existing_data_string:
                message = "Finetune data file already exists but is different at: {}".format(path)
                if strict:
                    raise Exception(message)
                else:
                    warnings.warn(message)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    return path


def load_finetune_data(platform_key: str, finetune_key: str) -> Optional[dict]:
    path = get_finetune_data_path(platform_key, finetune_key)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data
    else:
        return None


def list_of_dicts_to_dict_of_lists(list_of_dict):
    dict_of_lists = {}
    for key in list_of_dict[0].keys():
        dict_of_lists[key] = [d[key] for d in list_of_dict]
    return dict_of_lists
