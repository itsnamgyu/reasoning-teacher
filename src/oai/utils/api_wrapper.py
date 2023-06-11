import json
import os
import time
import traceback
from datetime import datetime

import openai

from oai.utils.metadata import FINETUNE_IDS_PATH, MODEL_IDS_PATH, set_model_id, get_model_key
from oai.utils.metadata import get_file_id, set_file_id, get_finetune_id, set_finetune_id
from paths import get_finetune_data_path, SAVED_PATH

OPENAI_ERROR_LOG_PATH = os.path.join(SAVED_PATH, "openai_error_log.txt")


def log_openai_error(message: str):
    timestamp = datetime.now().astimezone().isoformat()
    os.makedirs(os.path.dirname(OPENAI_ERROR_LOG_PATH), exist_ok=True)
    with open(OPENAI_ERROR_LOG_PATH, "a") as f:
        f.write(" {} ".format(timestamp).center(80, "#"))
        f.write("\n")
        f.write(message)
        f.write("\n")
        f.write("\n")


def get_openai_errors(lines=50):
    with open(OPENAI_ERROR_LOG_PATH) as f:
        if lines > 0:
            return "".join(f.readlines()[-lines:])
        else:
            return f.read()


def create_completion(*args, verbose=True, error_while=None, **kwargs):
    retry_intervals = [0] * 1 + [1] * 5 + [10, 30, 60, 300]

    for i, t in enumerate(retry_intervals):
        if t:
            time.sleep(t)
        try:
            response = openai.Completion.create(*args, **kwargs)
            return response
        except Exception as e:
            if verbose:
                print("Error during OpenAI completion attempt #{}: [{}] {}".format(i + 1, type(e).__name__, str(e)))
            if error_while is not None:
                log_openai_error("Error during {} attempt #{}:\n{}".format(error_while, i + 1, traceback.format_exc()))
            else:
                log_openai_error(traceback.format_exc())
    else:
        return None


def create_finetune_file(finetune_key: str, overwrite=False):
    if get_file_id(finetune_key) is not None and not overwrite:
        print("Warning: OpenAI File `{}` already exists (likely already uploaded). Skipping.".format(
            finetune_key))
        return

    path = get_finetune_data_path("openai", finetune_key)
    if not os.path.exists(path):
        raise FileNotFoundError("Finetune data file with file_key `{}` not found at: `{}`".format(finetune_key, path))

    with open(path) as f:
        response = openai.File.create(
            file=f,
            purpose='fine-tune'
        )
    file_id = response["id"]
    set_file_id(finetune_key, file_id)
    print("Created OpenAI File for `{}`: `{}`".format(finetune_key, response["id"]))

    return file_id


def create_finetune(file_key: str, base_model: str, dataset_key: str, train_key: str, ignore_existing=False,
                    **kwargs):
    model_key = get_model_key(base_model, dataset_key, train_key)
    if get_finetune_id(model_key) is not None and not ignore_existing:
        print("Warning: OpenAI Finetune for `{}` already exists. Skipping.".format(model_key))
        return

    file_id = get_file_id(file_key)
    if file_id is None:
        raise KeyError("OpenAI File with file_id `{}` does not exist".format(file_key))

    response = openai.FineTune.create(training_file=file_id, model=base_model, **kwargs)
    finetune_id = response["id"]
    set_finetune_id(model_key, finetune_id)
    print("Created OpenAI finetune `{}`: `{}`".format(model_key, finetune_id))

    return finetune_id


def fetch_model_ids():
    """
    Fetches model ids for all finetunes that have been completed
    """
    if os.path.exists(FINETUNE_IDS_PATH):
        with open(FINETUNE_IDS_PATH) as f:
            finetune_ids = json.load(f)
    else:
        raise FileNotFoundError(
            "Finetune ids metadata file is missing. Create a finetune using `oai.api_wrapper.create_finetune`")

    if os.path.exists(MODEL_IDS_PATH):
        with open(MODEL_IDS_PATH) as f:
            model_ids = json.load(f)
    else:
        model_ids = {}

    model_keys_to_fetch = []
    status_by_key = {}
    total = 0
    done = 0
    for model_key, finetune_id in finetune_ids.items():
        if model_key not in model_ids:
            model_keys_to_fetch.append(model_key)
            status_by_key[model_key] = "pending"
            total += 1

    if total == 0:
        print("No model ids to fetch")
        return True

    print("Fetching model ids from {} finetunes".format(len(model_keys_to_fetch)))
    print("-" * 100)
    print("{:<80s}{:<20s}".format("model_key", "status"))
    print("-" * 100)
    for model_key in model_keys_to_fetch:
        finetune_id = finetune_ids[model_key]
        response = openai.FineTune.retrieve(finetune_id)
        model_id = response["fine_tuned_model"]
        if model_id is not None:
            set_model_id(model_key, model_id)
            done += 1
        print("{:<80s}{:<20s}".format(model_key, response["status"]))
    print("-" * 100)
    print("Fetched {} of {} model ids".format(done, total))

    return done == total
