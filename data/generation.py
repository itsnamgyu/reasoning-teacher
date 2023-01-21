"""
Generate CoT samples from very large LLMs
"""
from typing import List, Optional

from tqdm import tqdm

from data.io import get_completion_data_path, save_completion_data, load_completion_data, load_dataset
from data.openai import create_completion
from data.tokens import truncate_by_n_tokens
from data.types import CompletionSample
from utils.metadata import get_model_id


def compose_zs_cot_reasoning_prompt(data_sample):
    q = data_sample["question"]
    a = data_sample["answer"]
    return "Q: {}\nA: Let's think step by step.".format(q, a)


def compose_zs_cot_answer_prompt(prompt, completion):
    return "{}{}\nTherefore, the answer is".format(prompt, completion)


def generate_cot_completions(completion_key, dataset_key, model_key, augmentations=None, reasoning_temperature=None,
                             indices: List[int] = None, request_batch_size=20, max_tokens=128,
                             max_tokens_used=None, step=None):
    if step is None:
        steps = ["reasoning", "answer"]
    else:
        if step not in {"reasoning", "answer"}:
            raise ValueError("Invalid argument for `step`: {}".format(step))
        steps = [step]

    try:
        assert max_tokens_used is None or isinstance(max_tokens_used, int) and 0 < max_tokens_used < max_tokens
    except AssertionError:
        raise ValueError(max_tokens_used)

    dataset = load_dataset(dataset_key)
    completion_data = load_completion_data(completion_key, dataset_key, model_key)
    model_id = get_model_id(model_key)

    if reasoning_temperature is None:
        if augmentations is None:
            reasoning_temperature = 0
        else:
            reasoning_temperature = 0.7

    if augmentations is None:
        augmentations = 1

    if indices is None:
        indices = list(range(len(dataset)))

    for i in indices:
        assert isinstance(i, int)
        assert i < len(dataset)

    if "reasoning" in steps:
        reasoning_candidates: List[CompletionSample] = []
        for index in indices:
            sample = dataset[index]
            for i in range(augmentations):
                try:
                    candidate = completion_data[index][i]  # note, this must refer to the object in `completion_data`
                    if "reasoning_completion" in candidate:
                        continue
                except IndexError:
                    pass
                # noinspection PyTypeChecker
                candidate: Optional[CompletionSample] = {
                    "sample_index": index,
                    "question": sample["question"],
                    "answer": sample["answer"],
                }
                reasoning_candidates.append(candidate)

        if len(reasoning_candidates) == 0:
            print("No additional reasoning completions needed")
        else:
            print(" Generating CoT Reasoning Completions (Step 1 - CoT Reasoning ) ".center(80, "-"))
            print("Saving to {}".format(get_completion_data_path(completion_key, dataset_key, model_key)))
            print("-" * 80)
            print("Completion:             {}".format(completion_key))
            print("Model:                  {}".format(model_key))
            print("Dataset:                {}".format(dataset_key))
            print("-" * 80)
            print("Samples:                {} of {}".format(len(indices), len(dataset)))
            print("Augmentations:          {}".format(augmentations))
            print("Reasoning Temperature:  {}".format(reasoning_temperature))
            print("Total Reasoning:        {}".format(len(indices) * augmentations))
            print("Remaining Reasoning:    {}".format(len(reasoning_candidates)))
            print("-" * 80)

            # Generate batches of completions at a time (due to limit by OpenAI)
            pbar = tqdm(list(range(len(reasoning_candidates))))
            candidate_batches = []
            for start in range(0, len(reasoning_candidates), request_batch_size):
                candidate_batches.append(reasoning_candidates[start:start + request_batch_size])
            for batch in candidate_batches:
                # Reasoning
                prompts = [compose_zs_cot_reasoning_prompt(sample) for sample in batch]
                response = create_completion(model=model_id, prompt=prompts, max_tokens=max_tokens,
                                             temperature=reasoning_temperature, n=1)
                completions = [c["text"] for c in response["choices"]]
                finish_reasons = [c["finish_reason"] for c in response["choices"]]
                for sample, prompt, completion, finish_reason in zip(batch, prompts, completions, finish_reasons):
                    sample["reasoning_prompt"] = prompt
                    sample["reasoning_completion"] = completion
                    sample["reasoning_finish_reason"] = finish_reason

                # Save
                for sample in batch:
                    if sample not in completion_data[sample["sample_index"]]:
                        completion_data[sample["sample_index"]].append(sample)
                save_completion_data(completion_data, completion_key, dataset_key, model_key)

                pbar.update(len(batch))

    if "answer" in steps:
        answer_candidates: List[CompletionSample] = []
        n_complete = 0
        n_candidates = 0
        n_no_reason = 0
        for index in indices:
            for i in range(augmentations):
                try:  # sample exists
                    candidate = completion_data[index][i]  # note, this must refer to the object in `completion_data`
                except IndexError:
                    candidate = None

                if candidate and "completion" in candidate:
                    n_complete += 1
                elif candidate and "reasoning_completion" in candidate:
                    n_candidates += 1
                    answer_candidates.append(candidate)
                else:  # no candidate or no reason
                    n_no_reason += 1

        if len(answer_candidates) == 0:
            print("No additional answer completions needed")
        else:
            print(" Generating CoT Reasoning Completions (Step 2 - Answers) ".center(80, "-"))
            print("Saving to {}".format(get_completion_data_path(completion_key, dataset_key, model_key)))
            print("-" * 80)
            print("Completion:             {}".format(completion_key))
            print("Model:                  {}".format(model_key))
            print("Dataset:                {}".format(dataset_key))
            print("-" * 80)
            print("Samples:                {} of {}".format(len(indices), len(dataset)))
            print("Augmentations:          {}".format(augmentations))
            print("Reasoning Temperature:  {}".format(reasoning_temperature))
            print("Total Answers:          {}".format(len(indices) * augmentations))
            print("Remaining Answers:      {}".format(n_candidates))
            print("Missing Reasons:        {}".format(n_no_reason))
            print("-" * 80)

            # Generate batches of completions at a time (due to limit by OpenAI)
            candidate_batches = []
            for start in range(0, len(answer_candidates), request_batch_size):
                candidate_batches.append(answer_candidates[start:start + request_batch_size])

            pbar = tqdm(list(range(len(answer_candidates))))
            for batch in candidate_batches:
                # Answer
                prompts = []
                for sample in batch:
                    p = sample["reasoning_prompt"]
                    if max_tokens_used is None:
                        c = sample["reasoning_completion"]
                    else:
                        c = truncate_by_n_tokens(sample["reasoning_completion"], n=max_tokens_used)
                    prompts.append(compose_zs_cot_answer_prompt(p, c))
                response = create_completion(model=model_id, prompt=prompts, max_tokens=128,
                                             temperature=0, n=1)
                completions = [c["text"] for c in response["choices"]]
                for sample, prompt, completion in zip(batch, prompts, completions):
                    sample["prompt"] = prompt
                    sample["completion"] = completion

                # Save
                for sample in batch:
                    if sample not in completion_data[sample["sample_index"]]:
                        completion_data[sample["sample_index"]].append(sample)
                save_completion_data(completion_data, completion_key, dataset_key, model_key)

                pbar.update(len(batch))
