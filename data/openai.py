import os
import time
import traceback
from datetime import datetime

import openai
from data.io import get_finetune_data_path
from utils.metadata import get_file_id, set_file_id, get_finetune_id, set_finetune_id
from utils.paths import SAVED_PATH

OPENAI_ERROR_LOG_PATH = os.path.join(SAVED_PATH, "openai_error_log.txt")


def log_openai_error(message: str):
    timestamp = datetime.now().astimezone().isoformat()
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
    times = [0] * 1 + [1] * 5 + [10, 30, 60, 300]
    for i, t in enumerate(times):
        if t:
            time.sleep(t)
        try:
            response = openai.Completion.create(*args, **kwargs)
            return response
        except Exception as e:
            if verbose:
                print("Attempt #{}: {}".format(i + 1, str(e)))
            if error_while is not None:
                log_openai_error("Error while: {}\n{}".format(error_while, traceback.format_exc()))
            else:
                log_openai_error(traceback.format_exc())
    else:
        return None


def create_finetune_file(file_key: str, ignore_existing=False):
    if get_file_id(file_key) is not None and not ignore_existing:
        print("Warning: finetune OpenAI file `{}` already exists (likely already uploaded). Skipping.".format(file_key))
        return

    path = get_finetune_data_path(file_key)
    if not os.path.exists(path):
        raise FileNotFoundError("Finetune data file with file_key `{}` not found at: `{}`".format(file_key, path))

    with open(path) as f:
        response = openai.File.create(
            file=f,
            purpose='fine-tune'
        )
    file_id = response["id"]
    set_file_id(file_key, file_id)
    print("Created OpenAI file for `{}`: `{}`".format(file_key, response["id"]))

    return file_id


def create_finetune(file_key: str, model_key: str, ignore_existing=False, **kwargs):
    if get_finetune_id(model_key) is not None and not ignore_existing:
        print("Warning: finetune for `{}` already exists. Skipping.".format(model_key))
        return

    file_id = get_file_id(file_key)
    if file_id is None:
        raise KeyError("File with file_id `{}` does not exist".format(file_key))

    response = openai.FineTune.create(training_file=file_id, **kwargs)
    finetune_id = response["id"]
    set_finetune_id(model_key, finetune_id)
    print("Created OpenAI finetune `{}`: `{}`".format(model_key, finetune_id))

    return finetune_id
