"""
CoT inference for finetuned models with one-step completion.
"""
from typing import List, Union

from tqdm import tqdm

from data.few_shot_cot import get_few_shot_cot_prompt, get_few_shot_cot_sample_indices
from data.io import load_dataset, get_completion_data_path, load_completion_data, save_completion_data
from data.openai import create_completion
from data.split import get_train_test_indices
from data.types import DatasetSample, CompletionSample
from utils.metadata import get_model_id


def compose_finetune_prompt(sample: Union[DatasetSample, CompletionSample], template="special"):
    q = sample["question"]
    if template is None:
        return "Q: {}\nA: The answer is".format(q)
    if template == "natural":
        return "Q: {}\n\nA: Let's think step by step.\n\n".format(q)
    if template == "special" or template is None:
        return "{}\n\n###\n\n".format(q)

    raise ValueError("Invalid template: {}".format(template))


def compose_few_shot_cot_prompt(dataset_key, sample: Union[DatasetSample, CompletionSample]):
    prompt = get_few_shot_cot_prompt(dataset_key)
    prompt += "\nQ: {}\nA:".format(sample["question"])
    return prompt


def infer_cot_completions(completion_key, dataset_key, model_key, template="special", split="test", split_seed=0,
                          train_ratio=None, augmentations=1, temperature=None, indices: List[int] = None,
                          stop_sequence="END", request_batch_size=20, max_tokens=128):
    """
    TODO: rename this to infer_completions
    TODO: merge this method with generate_completions
    :param completion_key: Default is `finetune_cot`
    :param dataset_key:
    :param model_key:
    :param template:
    :param split:
    :param split_seed:
    :param train_ratio:
    :param augmentations:
    :param temperature:
    :param indices:
    :param stop_sequence:
    :param request_batch_size:
    :param max_tokens:
    :return:
    """
    dataset = load_dataset(dataset_key)
    completion_data = load_completion_data(completion_key, dataset_key, model_key)
    model_id = get_model_id(model_key)

    if template == "few_shot_cot":
        stop_sequence = "Q:"

    if temperature is None:
        if augmentations == 1:
            temperature = 0
        else:
            temperature = 0.7

    if indices is None:
        if split is None:
            indices = list(range(len(dataset)))
        elif split in ["train", "test"]:
            train_indices, test_indices = get_train_test_indices(dataset_key, train_ratio=train_ratio,
                                                                 split_seed=split_seed)
            indices = train_indices if split == "train" else test_indices
            if split == "test" and template == "few_shot_cot":
                print("Automatically excluding cot samples from test set")
                indices = sorted(list(set(indices) - set(get_few_shot_cot_sample_indices(dataset_key))))
        else:
            raise ValueError("Invalid split: {}".format(split))

    if len(indices) == 0:
        print("Warning: `indices` is empty. Skipping.")
        return None
    for i in indices:
        assert isinstance(i, int) or print(i)
        assert i < len(dataset) or print(i)

    # Samples to generate (repeated for n_reasons)
    candidates: List[CompletionSample] = []
    for index in indices:
        sample = dataset[index]
        n_reasons = augmentations - len(completion_data[index])  # allow continued execution
        if n_reasons <= 0:
            continue
        for _ in range(n_reasons):
            # noinspection PyTypeChecker
            candidates.append({
                "sample_index": index,
                "question": sample["question"],
                "answer": sample["answer"],
            })

    if len(completion_data):
        print("Loaded {} existing completions".format(len(completion_data)))

    if len(candidates) == 0:
        print("No additional completions needed")
        return completion_data

    print(" Generating FT-CoT Completions ".center(80, "-"))
    completion_data_path = get_completion_data_path(completion_key, dataset_key, model_key)
    print(" ".center(80, "-"))
    print("Saving to {}".format(completion_data_path))
    print("-" * 80)
    print("Completion:             {}".format(completion_key))
    print("Dataset:                {}".format(dataset_key))
    print("Model:                  {}".format(model_key))
    print("-" * 80)
    print("Samples:                {} of {}".format(len(indices), len(dataset)))
    print("Augmentations:          {}".format(augmentations))
    print("Temperature:            {}".format(temperature))
    print("Total completions:      {}".format(len(indices) * augmentations))
    print("Remaining completions:  {}".format(len(candidates)))
    print("-" * 80)

    # Generate batches of completions at a time (due to limit by OpenAI)
    candidate_batches = []
    for start in range(0, len(candidates), request_batch_size):
        candidate_batches.append(candidates[start:start + request_batch_size])

    reassembled = []  # sanity check
    for b in candidate_batches:
        reassembled += b
    assert reassembled == candidates

    pbar = tqdm(list(range(len(candidates))))
    for batch in candidate_batches:
        # Reasoning
        if template == "few_shot_cot":
            prompts = [compose_few_shot_cot_prompt(dataset_key, sample) for sample in batch]
        else:
            prompts = [compose_finetune_prompt(sample, template) for sample in batch]
        response = create_completion(model=model_id, prompt=prompts, max_tokens=max_tokens,
                                     temperature=temperature, n=1, stop=stop_sequence)
        completions = [c["text"] for c in response["choices"]]
        finish_reasons = [c["finish_reason"] for c in response["choices"]]
        for sample, prompt, completion, finish_reason in zip(batch, prompts, completions, finish_reasons):
            sample["prompt"] = prompt
            sample["completion"] = completion
            sample["finish_reason"] = finish_reason

        # Save
        for sample in batch:
            completion_data[sample["sample_index"]].append(sample)
        save_completion_data(completion_data, completion_key, dataset_key, model_key)

        pbar.update(len(batch))

    return completion_data
