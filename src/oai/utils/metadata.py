import json
import os

from paths import SAVED_PATH

OPENAI_METADATA_PATH = os.path.join(SAVED_PATH, "openai_metadata")
FINETUNE_IDS_PATH = os.path.join(OPENAI_METADATA_PATH, "finetune_ids.json")
MODEL_IDS_PATH = os.path.join(OPENAI_METADATA_PATH, "model_ids.json")
FILE_IDS_PATH = os.path.join(OPENAI_METADATA_PATH, "file_ids.json")

DEFAULT_MODEL_IDS = [
    "ada",
    "babbage",
    "curie",
    "davinci",
    "text-davinci-001",
    "text-davinci-002",
    "text-davinci-003",
]


def get_json_value(json_path: str, key: str):
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return data.get(key, None)
    else:
        return None


def set_json_value(json_path: str, key: str, value=None, on_exist="overwrite"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    existing = data.get(key, None)
    if existing is None:
        if value is None:
            del data[key]
        else:
            data[key] = value
    else:
        if on_exist == "overwrite":
            if value is None:
                del data[key]
            else:
                data[key] = value
        elif on_exist == "check_equals":
            if existing != value:
                raise ValueError("Value mismatch: {} != {}".format(existing, value))
        elif on_exist == "ignore":
            pass
        else:
            raise ValueError("Unsupported argument for `on_exist`: {}".format(on_exist))

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def get_model_key(base_model_key: str, dataset_key: str, train_key: str):
    if train_key is None:
        return base_model_key
    else:
        return "B_{}__D_{}__T_{}".format(base_model_key, dataset_key, train_key)


def get_finetune_id(model_key):
    return get_json_value(FINETUNE_IDS_PATH, model_key)


def set_finetune_id(model_key, finetune_id, on_exist="overwrite"):
    set_json_value(FINETUNE_IDS_PATH, model_key, finetune_id, on_exist)


def get_model_id(model_key):
    if model_key in DEFAULT_MODEL_IDS:
        return model_key
    return get_json_value(MODEL_IDS_PATH, model_key)


def set_model_id(model_key, model_id, on_exist="overwrite"):
    if model_key in DEFAULT_MODEL_IDS:
        raise KeyError("Cannot overwrite default model: {}".format(model_key))
    set_json_value(MODEL_IDS_PATH, model_key, model_id, on_exist)


def get_file_id(finetune_key):
    return get_json_value(FILE_IDS_PATH, finetune_key)


def set_file_id(finetune_key, file_id, on_exist="overwrite"):
    set_json_value(FILE_IDS_PATH, finetune_key, file_id, on_exist)
