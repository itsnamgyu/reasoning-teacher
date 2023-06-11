import json
import os
from typing import List, Dict

from data.completion_dataset import CompletionDataset
from data.dataset import Dataset
from data.format import Formatter
from evaluation.evaluator import Evaluator
from oai.utils.api_wrapper import create_finetune_file, create_finetune
from oai.utils.metadata import get_model_key
from paths import get_finetune_data_path

STOP_PHRASE = "END"


def init_finetune(finetune_key: str, base_model: str, dataset_key: str, train_key: str,
                  finetune_kwargs: Dict = None) -> str:
    """
    Creates a `File` (containing the finetune data) and a `Finetune` (on that file) on OpenAI.

    Resulting `Model`s can be fetched after a Finetune is completed. Refer to `oai.utils.fetch_model_ids` to fetch
    models.

    Return model_key
    """
    create_finetune_file(finetune_key)
    model_key = get_model_key(base_model, dataset_key, train_key)
    if finetune_kwargs is None:
        finetune_kwargs = {}
    create_finetune(finetune_key, base_model, dataset_key, train_key, **finetune_kwargs)

    return model_key


def generate_finetune_data_from_completion_dataset(completion_dataset: CompletionDataset,
                                                   prediction_template: str,
                                                   finetune_key: str,
                                                   sample_indices: List[int] = None,
                                                   completion_indices: List[int] = None,
                                                   only_correct=True):
    """
    Generate
    """
    formatter = Formatter("decoder", prediction_template, dataset_key=completion_dataset.dataset_key,
                          stop_phrase=STOP_PHRASE)
    samples = completion_dataset.select_samples(sample_indices, completion_indices)

    if only_correct:
        evaluator = Evaluator(completion_dataset.dataset_key, completion_dataset.prediction_template)

    finetune_data = []
    for sample in samples:
        if only_correct:
            if not evaluator.evaluate_completion(sample)["correct"]:
                continue
        formatted = formatter(sample, include_label=True)
        finetune_data.append({
            "prompt": formatted["input"],
            "completion": formatted["label"],
        })

    _save_finetune_data(finetune_data, finetune_key)


def generate_finetune_data_from_dataset(dataset: Dataset,
                                        prediction_template: str,
                                        finetune_key: str,
                                        sample_indices: List[int] = None):
    formatter = Formatter("decoder", prediction_template, dataset_key=dataset.dataset_key,
                          stop_phrase=STOP_PHRASE)
    samples = dataset.select_samples(sample_indices)

    finetune_data = []
    for sample in samples:
        formatted = formatter(sample, include_label=True)
        finetune_data.append({
            "prompt": formatted["input"],
            "completion": formatted["label"],
        })

    _save_finetune_data(finetune_data, finetune_key)


def _save_finetune_data(data: List[Dict], finetune_key):
    path = get_finetune_data_path("openai", finetune_key)
    print("Saving {} fine-tuning samples to {}".format(len(data), path))

    lines = []
    for sample in data:
        lines.append(json.dumps(sample))
    full_string = "\n".join(lines)

    if os.path.exists(path):
        with open(path, "r") as f:
            existing = f.read()
        if existing != full_string:
            raise Exception("Finetune data already exists and is different from the given data.")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(full_string)
