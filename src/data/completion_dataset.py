"""
CompletionDataset class and related functions to load and save generated completions from openai or custom models.

Format: {
    "metadata": {
        ...
    },
    "data": {
        sample_index: [  # ← list of "completion samples" (prediction) for each "sample" (question in the dataset).
            {  # ← completion sample dict
                "sample_index": int,
                "completion_index": int,
                "question": str,
                "answer": str,
                "reasoning_prompt": str, # for zero-shot-cot
                "reasoning_completion": str,  # for zero-shot-cot
                "reasoning_finish_reason": str,  # for openai
                "prompt": str,
                "completion": str,
                "finish_reason": str,  # for openai
            },
            ...
        ],
        ...
    }
}
"""

import json
import os
from typing import Dict, List

from easydict import EasyDict

from paths import get_completion_data_path


class CompletionIdentifier:
    """
    Shorthand for CompletionDatasetIdentifier. Contains all information to identify a completion dataset, i.e., the
    path of the finetune data file.
    """

    def __init__(self, base_model: str, completion_key: str, dataset_key: str,
                 train_key: str = None, epoch: int = None):
        self.base_model = base_model
        self.completion_key = completion_key
        self.dataset_key = dataset_key
        self.train_key = train_key
        self.epoch = epoch

    def __repr__(self):
        return "{}_{}_{}_{}_{}".format(self.base_model, self.completion_key, self.dataset_key,
                                       "NAN" if self.train_key is None else self.train_key,
                                       "NAN" if self.epoch is None else self.epoch)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other: "CompletionIdentifier"):
        return self.base_model == other.base_model and self.completion_key == other.completion_key and \
            self.dataset_key == other.dataset_key and self.train_key == other.train_key and \
            self.epoch == other.epoch

    @property
    def data_path(self):
        return get_completion_data_path(self.base_model, self.completion_key, self.dataset_key,
                                        self.train_key, self.epoch)


class CompletionMetadata(CompletionIdentifier):
    """
    Contains the minimum metadata that constitutes a valid CompletionDataset. Uses include
    - passing completion information to a LightningModule to generate validation data
    """

    def __init__(self, base_model: str, completion_key: str, dataset_key: str, finetune_key: str = None,
                 prediction_template: str = None, train_key: str = None, epoch: int = None):
        super().__init__(base_model, completion_key, dataset_key, train_key, epoch)
        self.finetune_key = finetune_key
        self.prediction_template = prediction_template


class CompletionDataset:
    def __init__(self, raw_data: Dict):
        self.metadata: EasyDict = EasyDict(raw_data["metadata"])
        self.data: Dict[int, List] = {int(k): v for k, v in raw_data["data"].items()}

    @property
    def base_model(self):
        return self.metadata["base_model"]

    @property
    def dataset_key(self):
        return self.metadata["dataset_key"]

    @property
    def completion_key(self):
        return self.metadata["completion_key"]

    @property
    def finetune_key(self):
        return self.metadata["finetune_key"]

    @property
    def train_key(self):
        return self.metadata["train_key"]

    @property
    def epoch(self):
        return self.metadata["epoch"]

    @property
    def prediction_template(self):
        return self.metadata["prediction_template"]

    def __len__(self):
        return len(self.data)

    @property
    def total_samples(self):
        return sum([len(v) for v in self.data.values()])

    def __getitem__(self, sample_index: int) -> List:
        return self.data[sample_index]

    @property
    def indices(self):
        return list(self.data.keys())

    def select_samples(self, sample_indices: List[int] = None, completion_indices: List[int] = None,
                       only_correct=False) -> List[dict]:
        """
        Filter and retrieve completions based on given indices and correctness.

        - sample_indices: List of sample indices to select completions from.
        - completion_indices: List of completion indices to select for each sample
        - only_correct: If True, only correct completions are returned. Used for CoT dataset generation.

        Return: List of completions.
        """
        if sample_indices is None:
            sample_indices = list(self.data.keys())
        else:
            unavailable = list(set(sample_indices) - set(self.data.keys()))
            if unavailable:
                raise Exception("Unavailable sample indices including {}".format(unavailable[:5]))

        if only_correct:
            evaluator = self.get_evaluator()

        completions = []
        for s in sample_indices:
            candidates = []
            samples = self.data[s]
            if completion_indices is None:  # add all completions
                candidates += samples
            else:  # check if completions exist and add
                for c in completion_indices:
                    if len(samples) <= c:
                        raise ValueError("Completion #{} for sample #{} not found.".format(c, s))
                    candidates += [samples[c]]
            for c in candidates:
                if not only_correct or evaluator.check_answer(c["completion"], c["answer"]):
                    completions.append(c)

        return completions

    @property
    def path(self):
        return get_completion_data_path(self.base_model, self.completion_key, self.dataset_key, self.train_key,
                                        self.epoch)

    def save(self):
        raw_data = {
            "metadata": self.metadata,
            "data": self.data
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(raw_data, f, indent=4)

    @staticmethod
    def init(completion_metadata: CompletionMetadata, additional_metadata: Dict = None):
        raw_data = {
            "metadata": {
                "base_model": completion_metadata.base_model,
                "dataset_key": completion_metadata.dataset_key,
                "completion_key": completion_metadata.completion_key,
                "finetune_key": completion_metadata.finetune_key,
                "train_key": completion_metadata.train_key,
                "epoch": completion_metadata.epoch,
                "prediction_template": completion_metadata.prediction_template,
            },
            "data": {}
        }
        if additional_metadata:
            raw_data["metadata"].update(additional_metadata)
        return CompletionDataset(raw_data)

    @staticmethod
    def load(completion_identifier: CompletionIdentifier):
        base_model = completion_identifier.base_model
        dataset_key = completion_identifier.dataset_key
        completion_key = completion_identifier.completion_key
        train_key = completion_identifier.train_key
        epoch = completion_identifier.epoch

        with open(get_completion_data_path(base_model, completion_key, dataset_key, train_key, epoch)) as f:
            raw_data = json.load(f)
        completions = CompletionDataset(raw_data)

        if completions.base_model != base_model:
            raise Exception("Base model mismatch.")
        if completions.dataset_key != dataset_key:
            raise Exception("Dataset key mismatch.")
        if completions.completion_key != completion_key:
            raise Exception("Completion key mismatch.")
        if completions.train_key != train_key:
            raise Exception("Train key mismatch.")
        if completions.epoch != epoch:
            raise Exception("Epoch mismatch.")

        return completions

    def get_evaluator(self):
        from evaluation.evaluator import Evaluator
        return Evaluator(self.dataset_key, self.metadata["prediction_template"])
