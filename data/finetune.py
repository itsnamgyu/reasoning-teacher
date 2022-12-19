import json
import warnings
from typing import Union, List

from data.evaluation import compare_prediction_and_answer, cleanse_prediction, ANSWER_PREFIXES, cleanse_answer
from data.io import save_finetune_data
from data.split import get_train_test_indices
from data.types import CompletionSample, DatasetSample


def compose_finetune_sample(sample: Union[DatasetSample, CompletionSample], template="special"):
    q = sample["question"]
    a = sample["answer"]

    # Dataset sample used for ft (vanilla)
    if template is None:
        return {
            "prompt": "{}\n\n ### \n\n".format(q),
            "completion": " {} END".format(a),
        }

    # Completion sample used for ft_cot
    reasoning = sample["reasoning_completion"].strip()
    if template == "natural":
        return {
            "prompt": "Q: {}\n\nA: Let's think step by step.\n\n".format(q),
            "completion": " {}\n\nTherefore, the answer is\n\n{}. END".format(reasoning, a),
        }
    if template == "special":
        return {
            "prompt": "{}\n\n###\n\n".format(q),
            "completion": " {}\n\n-->\n\n{} END".format(reasoning, a),
        }

    raise ValueError("Invalid format {}".format(template))


def generate_finetune_data(data: Union[List[DatasetSample], List[CompletionSample]], dataset_key, template="natural",
                           stop_sequence="END", file_key=None, ignore_existing=False, indices=None, augmentations=1):
    is_dataset_sample = (template is None)

    if is_dataset_sample and augmentations != 1:
        error = "Invalid combination: `template=None` and `augmentatios != 1`. " \
                "There are no augmentations for non-reasoning samples."
        raise ValueError(error)

    if indices is None:  # assert that completion_data comprises train_indices *only*
        train_indices, test_indices = get_train_test_indices(dataset_key)
        assert_message = "Expected completion data to be comprised of, and only of, train samples"
        assert set(train_indices) == set(data.keys()) or print(assert_message)

    if augmentations > 1 and indices is None:
        warning = "There are no experiments that require multiple augmentations and all indices. We recommend that " \
                  "you specify `indices`"
        warnings.warn(warning)

    if indices is None:
        indices = list(data.keys())

    lines = []
    for index in indices:
        if is_dataset_sample:
            samples = [data[index]]
        else:
            samples = data[index][:augmentations]
            if len(samples) < augmentations:
                error = "Insufficient number of samples ({} of {}) for sample #{}".format(
                    len(samples), augmentations, index)
                raise Exception(error)

        for sample in samples:
            if not is_dataset_sample:
                answer_prefix = ANSWER_PREFIXES[template]
                prediction = cleanse_prediction(sample["completion"], dataset_key, answer_prefix=answer_prefix,
                                                stop_sequence=stop_sequence, return_all=False)
                answer = cleanse_answer(sample["answer"], dataset_key)
                is_correct_completion = compare_prediction_and_answer(prediction, answer, dataset_key)
            if is_dataset_sample or is_correct_completion:
                s = compose_finetune_sample(sample, template)
                lines.append(json.dumps(s))

    if len(lines) == 0:
        print("Warning: No data to generate. Skipping.")
        return

    finetune_data = "\n".join(lines)
    if file_key is not None:
        save_finetune_data(finetune_data, file_key, ignore_existing)

    return finetune_data
