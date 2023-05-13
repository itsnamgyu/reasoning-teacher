"""
Preprocess completion data into a format that can be used to fine-tune custom (local) models such as T5.
"""

from collections import defaultdict

from data.completion_dataset import CompletionDataset
from data.dataset import Dataset
from data.format import Formatter
from data.split import load_split


def compile_cot_train_data(completion_dataset: CompletionDataset, model_type, sample_indices=None,
                           completion_indices=None, only_correct=True):
    if "zs_cot" not in completion_dataset.metadata["completion_key"]:
        raise ValueError("Completion key {} not implemented.".format(completion_dataset.metadata["completion_key"]))

    completions = completion_dataset.select_samples(sample_indices, completion_indices,
                                                    only_correct)
    formatter = Formatter(model_type, "ft_cot_token")
    formatted = formatter.format_samples(completions, include_labels=True)

    data = defaultdict(list)
    for sample in formatted:
        for key in ["sample_index", "input", "label"]:
            data[key].append(sample[key])
    return dict(data)


def compile_cot_test_data(dataset_key: str, model_type, split_key="default", subset_key="test"):
    dataset = Dataset.load(dataset_key)
    indices = load_split(dataset_key, split_key)[subset_key]
    samples = dataset.select_samples(indices)

    formatter = Formatter(model_type, "ft_cot_token")
    formatted = formatter.format_samples(samples, include_labels=False)

    data = defaultdict(list)
    for sample in formatted:
        for key in ["sample_index", "input"]:
            data[key].append(sample[key])
    return dict(data)
