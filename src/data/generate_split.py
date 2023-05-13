"""
This script is used to generate splits. This should only be used for initial generation (by the author, or contributors
who add new dataset or splits). Subsequent runs should use the splits saved in `data/splits/` to ensure consistency.

Note, `np.random.RandomState` is said to guarantee consistency across environments, but we do this just to be safe.
"""
import json
import os
from typing import Optional, Tuple, List

import numpy as np

from data.dataset import DATASET_SIZES
from paths import get_split_path


def get_default_train_test_split(dataset_key) -> Optional[Tuple[List[int], List[int]]]:
    predefined = get_predefined_train_test_split(dataset_key)
    if predefined is not None:
        return predefined

    return get_random_train_test_indices(dataset_key)


def get_predefined_train_test_split(dataset_key) -> Optional[Tuple[List[int], List[int]]]:
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


def get_random_train_test_indices(dataset_key: str, train_ratio=0.7, split_seed=0) -> Tuple[
    List[int], List[int]]:
    dataset_size = DATASET_SIZES[dataset_key]

    indices = list(range(dataset_size))
    state = np.random.RandomState(split_seed)
    indices = state.permutation(indices)
    train_n = round(dataset_size * train_ratio)
    train_indices = sorted(indices[:train_n].tolist())
    test_indices = sorted(indices[train_n:].tolist())

    return train_indices, test_indices


if __name__ == "__main__":
    for dataset_key in DATASET_SIZES:
        path = get_split_path(dataset_key, "default")
        if os.path.exists(path):
            print("Skipping `{}`. Split file already exists at: {}".format(dataset_key, path))
            continue

        train, test = get_default_train_test_split(dataset_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"train": train, "test": test}, f, indent=4)
        print("Saved split for `{}` to: {}".format(dataset_key, path))
