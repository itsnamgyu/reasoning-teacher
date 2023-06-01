"""
Dataset class to load original benchmark datasets

Format: {
    "metadata": {
        "dataset_key": str,
    },
    "data": [
        {
            "sample_index": int,
            "question": str,
            "answer": str,
        },
        ...
    ]
}
"""

import json
from typing import Dict, List

from easydict import EasyDict

from paths import get_dataset_path

DATASET_SIZES = {
    "single_eq": 508,
    "addsub": 395,
    "multiarith": 600,
    "gsm8k": 8792,
    "aqua": 97721,
    "svamp": 1000,

    "tracking_shuffled_objects": 750,
    "date_understanding": 369,
    "coin_flip": 500,
    "last_letter_concatenation": 500,

    "commonsense_qa": 10962,
    "strategy_qa": 2290,
}

DATASET_KEYS = list(DATASET_SIZES.keys())


class Dataset:
    def __init__(self, raw_data: Dict):
        self.raw_data = raw_data
        self.metadata: EasyDict = EasyDict(raw_data["metadata"])
        self.data: List[Dict] = raw_data["data"]

    @property
    def dataset_key(self):
        return self.metadata["dataset_key"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int) -> Dict:
        return self.data[item]

    @staticmethod
    def load(dataset_key):
        with open(get_dataset_path(dataset_key), "r") as f:
            raw_data = json.load(f)
        dataset = Dataset(raw_data)

        if dataset.metadata["dataset_key"] != dataset_key:
            raise Exception("Dataset key mismatch.")

        return dataset

    def select_samples(self, sample_indices: List[int]) -> List[Dict]:
        selected = []
        for i in sample_indices:
            if len(self.data) <= i:
                raise IndexError("Sample index {} out of range [0, {}).".format(i, len(self.data)))
            selected.append(self.data[i])
        return selected
