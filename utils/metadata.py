import json
import os

from utils.paths import SAVED_PATH

FINETUNE_IDS_PATH = os.path.join(SAVED_PATH, "finetune_ids.json")
MODEL_IDS_PATH = os.path.join(SAVED_PATH, "model_ids.json")
FILE_IDS_PATH = os.path.join(SAVED_PATH, "file_ids.json")


def set_finetune_id(finetune_key, finetune_id):
    if os.path.exists(FINETUNE_IDS_PATH):
        with open(FINETUNE_IDS_PATH) as f:
            finetune_ids = json.load(f)
    else:
        finetune_ids = {}

    existing = finetune_ids.get(finetune_key, None)
    if existing is not None and existing != finetune_id:
        print("Warning: overriding finetune_id for {}".format(finetune_key))
        print("From:", existing)
        print("To  :", finetune_id)
    finetune_ids[finetune_key] = finetune_id

    with open(FINETUNE_IDS_PATH, "w") as f:
        json.dump(finetune_ids, f, indent=4)


def get_finetune_id(finetune_key):
    if os.path.exists(FINETUNE_IDS_PATH):
        with open(FINETUNE_IDS_PATH) as f:
            finetune_ids = json.load(f)
        return finetune_ids.get(finetune_key, None)
    return None


def set_model_id(model_key, model_id, ignore_existing=False):
    if os.path.exists(MODEL_IDS_PATH):
        with open(MODEL_IDS_PATH) as f:
            model_ids = json.load(f)
    else:
        model_ids = {}

    existing = model_ids.get(model_key, None)
    if existing is not None and existing != model_id:
        if ignore_existing:
            print("Warning: overriding model_id for {}".format(model_key))
            print("From:", existing)
            print("To  :", model_id)
        else:
            print("model_id for {} already exists. Skipping.".format(model_key))
            return

    model_ids[model_key] = model_id
    with open(MODEL_IDS_PATH, "w") as f:
        json.dump(model_ids, f, indent=4)

    return


def get_model_id(model_key, strict=True):
    if os.path.exists(MODEL_IDS_PATH):
        with open(MODEL_IDS_PATH) as f:
            model_ids = json.load(f)
        model_id = model_ids.get(model_key, None)
        if model_id is None and strict:
            raise ValueError("No model_id found for model_key `{}`".format(model_key))
        return model_id
    return None


def set_file_id(file_key, file_id, ignore_existing=False):
    if os.path.exists(FILE_IDS_PATH):
        with open(FILE_IDS_PATH) as f:
            file_ids = json.load(f)
    else:
        file_ids = {}

    existing = file_ids.get(file_key, None)
    if existing is not None and existing != file_id:
        if ignore_existing:
            print("Warning: overriding file_id for {}".format(file_key))
            print("From:", existing)
            print("To  :", file_id)
        else:
            print("file_id for {} already exists. Skipping.".format(file_key))
            return

    file_ids[file_key] = file_id
    with open(FILE_IDS_PATH, "w") as f:
        json.dump(file_ids, f, indent=4)


def get_file_id(file_key):
    if os.path.exists(FILE_IDS_PATH):
        with open(FILE_IDS_PATH) as f:
            file_ids = json.load(f)
        return file_ids.get(file_key, None)
    return None
