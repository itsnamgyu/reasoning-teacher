import json
import os
from typing import List, Tuple

import numpy as np

from paths import get_split_path


def load_train_test_split(dataset_key: str) -> Tuple[List[int], List[int]]:
    split = load_split(dataset_key)
    return split["train"], split["test"]


def load_split(dataset_key: str, split_key="default"):
    path = get_split_path(dataset_key, split_key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError("Split {} for dataset {} does not exist.".format(split_key, dataset_key))


def subsample_indices(indices: List[int], n: int, split_seed=0):
    """
    Sort, permute and select first `n` indices.
    Used for 8, 32, 128shot ablations in paper.
    """
    if n > len(indices):
        print("Warning: n == {} > len(indices) == {}".format(n, len(indices)))

    state = np.random.RandomState(split_seed)
    indices = state.permutation(indices)
    return sorted(indices[:n].tolist())
