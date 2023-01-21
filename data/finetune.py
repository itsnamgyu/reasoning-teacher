import json
import warnings
from typing import Union, List, Dict

from data.evaluation import compare_prediction_and_answer, cleanse_prediction, ANSWER_PREFIXES, cleanse_answer
from data.io import save_finetune_data
from data.split import get_train_test_indices
from data.types import CompletionSample, DatasetSample


def compose_finetune_sample(sample: Union[DatasetSample, CompletionSample], template="special"):
    q = sample["question"]
    a = sample["answer"]

    # Dataset sample used for ft (vanilla)
    if template is None or template == "vanilla_special":
        return {
            "prompt": "{}\n\n ### \n\n".format(q),
            "completion": " {} END".format(a),
        }
    if template == "custom_t5_vanilla_special":
        return {
            "prompt": "{}".format(q),
            "completion": a,
        }
    if template == "custom_generative_vanilla_special":
        return {
            "prompt": "{}\n\n###\n\n".format(q),
            "completion": a,
        }

    # Completion sample used for ft_cot
    reasoning = sample["reasoning_completion"].strip()
    if template == "natural":
        return {
            "prompt": "Q: {}\n\nA: Let's think step by step.\n\n".format(q),
            "completion": " {}\n\nTherefore, the answer is\n\n{} END".format(reasoning, a),
        }
    if template == "special":
        return {
            "prompt": "{}\n\n###\n\n".format(q),
            "completion": " {}\n\n-->\n\n{} END".format(reasoning, a),
        }

    if template == "custom_t5_natural":
        return {
            "prompt": "solve step by step: {}".format(q),
            "completion": "{}\n\nTherefore, the answer is {}".format(reasoning, a),
        }
    if template == "custom_t5_special":
        return {
            "prompt": "{}".format(q),
            "completion": "{} --> {}".format(reasoning, a),
        }
    if template == "custom_generative_natural":
        return {
            "prompt": "Q: {}\n\nA: Let's think step by step.\n\n".format(q),
            "completion": "{}\n\nTherefore, the answer is\n\n{}.".format(reasoning, a),
        }
    if template == "custom_generative_special":
        return {
            "prompt": "{}\n\n###\n\n".format(q),
            "completion": "{}\n\n-->\n\n{}".format(reasoning, a),
        }

    raise ValueError("Invalid format {}".format(template))


def generate_finetune_data(data: Union[Dict[int, DatasetSample], Dict[int, List[CompletionSample]]], dataset_key,
                           template="natural", file_key=None, ignore_existing=False, indices=None, augmentations=1,
                           max_samples=None, augmentation_indices: Dict[int, List[int]] = None, include_incorrect=False):
    """
    :param data:
    :param dataset_key:
    :param template:
    :param file_key:
    :param ignore_existing:
    :param indices:
    :param augmentations:
    :param max_samples: Maximum number of fine-tune samples to generate per original training sample. This is
    used for the efficient diverse reasoning experiment, which uses *up to* N=`max_samples` correct diverse reasoning
    samples.
    :param augmentation_indices:
    :param include_incorrect:
    :return:
    """
    is_dataset_sample = template is None or "vanilla" in template  # HOTFIX

    if is_dataset_sample and augmentations != 1:
        error = "Invalid combination: `template=None` and `augmentations != 1`. " \
                "There are no augmentations for non-reasoning samples."
        raise ValueError(error)

    if indices is None:  # assert that completion_data comprises train_indices *only*, unless otherwise specified
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
            if augmentation_indices:
                # Fetch augmentations corresponding to `augmentation_indices[index]`
                samples = []
                if index not in augmentation_indices:
                    raise ValueError("`augmentation_indices` keys must include all sample indices")
                for aug_index in augmentation_indices[index]:
                    try:
                        samples.append(data[index][aug_index])
                    except IndexError:
                        error = "Invalid augmentation index {} (total is {}) for sample #{}".format(
                            aug_index, len(sample), index)
                        raise IndexError(error)
            else:
                # Use first `augmentations`
                samples = data[index][:augmentations]
                if len(samples) < augmentations:
                    error = "Insufficient number of samples ({} of {}) for sample #{}".format(
                        len(samples), augmentations, index)
                    raise Exception(error)

        added = 0
        for sample in samples:
            if max_samples is not None and added >= max_samples:
                break
            if not is_dataset_sample:
                answer_prefix = ANSWER_PREFIXES[template]
                prediction = cleanse_prediction(sample["completion"], dataset_key, answer_prefix=answer_prefix,
                                                return_all=False)
                answer = cleanse_answer(sample["answer"], dataset_key)
                is_correct_completion = compare_prediction_and_answer(prediction, answer, dataset_key)
            if is_dataset_sample or is_correct_completion or include_incorrect:
                s = compose_finetune_sample(sample, template)
                lines.append(json.dumps(s))
                added += 1

    if len(lines) == 0:
        print("Warning: No data to generate. Skipping.")
        return

    finetune_data = "\n".join(lines)
    if file_key is not None:
        save_finetune_data(finetune_data, file_key, ignore_existing)

    return finetune_data
