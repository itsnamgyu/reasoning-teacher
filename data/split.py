import json
import os
from typing import Tuple, List, Optional

import numpy as np

from data.io import DATASET_SIZES
from utils.paths import CUSTOM_SPLIT_PATH

DEFAULT_TRAIN_RATIO = 0.7


def get_default_train_test_split(dataset_key) -> Optional[Tuple[List[int], List[int]]]:
    if dataset_key == "aqua":
        train_size = 97467
        test_size = 254
        train_subsample_seed = 0
        train_subsample_size = 10000
        train_indices = list(range(train_size))
        state = np.random.RandomState(train_subsample_seed)
        train_indices = sorted(state.permutation(train_indices)[:train_subsample_size].tolist())
        test_indices = list(range(train_size, train_size + test_size))
        return train_indices, test_indices
    if dataset_key == "gsm8k":
        train_size = 7473
        test_size = 1319
        indices = list(range(train_size + test_size))
        return indices[:train_size], indices[train_size:]
    if dataset_key == "commonsense_qa":
        train_size = 9741
        test_size = 1221
        indices = list(range(train_size + test_size))
        return indices[:train_size], indices[train_size:]

    return None


def get_custom_train_test_split(dataset_key, split_key) -> Optional[Tuple[List[int], List[int]]]:
    custom_split_path = os.path.join(CUSTOM_SPLIT_PATH, "{}_{}.json".format(dataset_key, split_key))
    print("Loading custom split from: `{}`".format(custom_split_path))
    with open(custom_split_path) as f:
        split = json.load(f)
    return split["train"], split["test"]


def get_train_test_indices(dataset_key: str, train_ratio: float = None, split_key=None, split_seed=0):
    """
    :param dataset_key:
    :param train_ratio: If none, default predefined split will be used if defined, else defaults to DEFAULT_TRAIN_RATIO
    :param split_key: Use custom predefined split saved in CUSTOM_SPLIT_PATH
    :param split_seed: Used for split based on train_ratio
    :return:
    """
    dataset_size = DATASET_SIZES[dataset_key]

    if train_ratio is not None and split_key is not None:
        raise ValueError("Cannot use both `train_ratio` and `split_key")

    if split_key is not None:  # use custom predefined split
        return get_custom_train_test_split(dataset_key, split_key)

    if train_ratio is None:  # use default predefined split, if it exists
        default_split = get_default_train_test_split(dataset_key)
        if default_split is not None:
            train_indices, test_indices = default_split
            assert len(set(train_indices + test_indices)) == len(train_indices + test_indices)
            assert len(set(train_indices).intersection(set(test_indices))) == 0
            assert set(train_indices + test_indices).issubset(set(range(dataset_size)))

            train_indices.sort(), test_indices.sort()
            return train_indices, test_indices
        else:  # most default: split by default train ratio
            train_ratio = DEFAULT_TRAIN_RATIO

    # split by given train_ratio
    indices = list(range(dataset_size))
    state = np.random.RandomState(split_seed)
    indices = state.permutation(indices)
    train_n = round(dataset_size * train_ratio)
    train_indices = sorted(indices[:train_n].tolist())
    test_indices = sorted(indices[train_n:].tolist())

    return train_indices, test_indices


def get_few_shot_train_indices(dataset_key: str, shots=8, train_ratio=None, split_seed=0):
    """
    Permute *sorted train_indices* and select first `shots` indices

    :param dataset_key:
    :param shots:
    :param train_ratio:
    :param split_seed:
    :return:
    """
    train_indices, test_indices = get_train_test_indices(dataset_key, train_ratio=train_ratio, split_seed=split_seed)
    if shots > len(train_indices):
        print("Warning: number of shots {} > number of training samples {}. Using all training samples".format(
            shots, len(train_indices)))

    state = np.random.RandomState(split_seed)
    train_indices = state.permutation(train_indices)
    few_shot_indices = sorted(train_indices[:shots].tolist())

    return few_shot_indices
