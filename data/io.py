"""
Dataset: [
    {
        question: str,
        answer: str,
    }, ...
]

Completion Data: {
    0: [
        question: str,
        answer: str,
        prompt: str,
        completion: str,
    ]
}
"""
import json
import os
from collections import defaultdict
from json import JSONDecodeError
from typing import Dict

from utils.paths import COMPLETION_DATA_PATH, DATASET_PATH, FINETUNE_DATA_PATH


DATASET_SIZES = {
    "single_eq": 508,
    "addsub": 395,
    "multiarith": 600,
    "gsm8k": 8792,
    "gsm8k_test_only": 1319,
    "aqua": 97721,
    "svamp": 1000,

    "tracking_shuffled_objects": 750,
    "date_understanding": 369,
    "coin_flip": 500,
    "last_letter_concatenation": 500,

    "commonsense_qa": 10962,
    "strategy_qa": 2290,
}


def get_dataset_path(dataset_key):
    return os.path.join(DATASET_PATH, "{}.json".format(dataset_key))


def load_dataset(dataset_key) -> Dict:
    with open(get_dataset_path(dataset_key)) as f:
        dataset = json.load(f)
    if dataset_key not in DATASET_SIZES:
        raise Exception(
            "Dataset `{}` not specific in DATASET_SIZES. Please add manually for cross-validation.".format(dataset_key))

    documented_size = DATASET_SIZES[dataset_key]
    actual_size = len(dataset)

    if actual_size != documented_size:
        message = "Invalid dataset size for `{}`. Documented size / actual size: {} / {}".format(
            dataset_key, documented_size, actual_size)
        raise Exception(message)
    return dataset


def get_completion_data_path(completion_key, dataset_key, model_key):
    basename = "{}_{}_{}.json".format(completion_key, dataset_key, model_key)
    return os.path.join(COMPLETION_DATA_PATH, basename)


def load_completion_data(completion_key, dataset_key, model_key) -> defaultdict:
    """
    Returns completion data with sample indices type casted into integers
    :param completion_key:
    :param dataset_key:
    :param model_key:
    :return:
    """
    path = get_completion_data_path(completion_key, dataset_key, model_key)
    completion_data = defaultdict(list)
    if os.path.exists(path):
        with open(path) as f:
            loaded = json.load(f)
        for key, value in loaded.items():
            completion_data[int(key)] = value
    return completion_data


def save_completion_data(completion_data, completion_key, dataset_key, model_key) -> str:
    path = get_completion_data_path(completion_key, dataset_key, model_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(completion_data, f, indent=4)
    return path


def check_finetune_data(finetune_data):
    for line in finetune_data.split("\n"):
        try:
            json.loads(line)
        except JSONDecodeError as e:
            raise AssertionError("finetune_data should be in jsonl format (i.e., lines of json strings)")


def get_finetune_data_path(file_key):
    return os.path.join(FINETUNE_DATA_PATH, "{}.json".format(file_key))


def save_finetune_data(finetune_data: str, file_key: str, ignore_existing=False):
    check_finetune_data(finetune_data)
    path = get_finetune_data_path(file_key)

    if not ignore_existing and os.path.exists(path):
        print("Warning: finetune file `{}` already exists at `{}`. Skipping.".format(file_key, path))
        return None

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(finetune_data)
        print("Saved {} finetune samples for `{}` to:".format(finetune_data.count("\n") + 1, file_key))
        print(path)
    return path
