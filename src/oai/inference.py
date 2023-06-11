from typing import List, Dict

from tqdm import tqdm

from data.completion_dataset import CompletionMetadata, CompletionDataset
from data.dataset import Dataset
from data.format import Formatter
from oai.utils.api_wrapper import create_completion
from oai.utils.metadata import get_model_key, get_model_id

STOP_PHRASE = "END"


def batch_infer_samples(samples: List[Dict], model_id: str, key_prefix: None, batch_size: int = 20,
                        temperature: float = 0, max_tokens: int = 128,
                        save_completion_dataset: CompletionDataset = None):
    """
    Complete samples using OpenAI models in batches, in-place.

    - All samples should contain a prompt with the key "<key_prefix>_prompt".
    - Completions will be added with the key "<key_prefix>_completion".
    - finish_reason's will be added with the key "<key_prefix>_finish_reason".

    - saved_completion_dataset: if provided, will be saved every time a batch is completed. `samples` should
      contain references to the samples in this CompletionDataset, or else the new completions will not be saved.
    """
    # Prepend key_prefix to keys, e.g., "reasoning" for zs_cot step 1
    prompt_key = f"{key_prefix}_prompt" if key_prefix else "prompt"
    completion_key = f"{key_prefix}_completion" if key_prefix else "completion"
    finish_reason_key = f"{key_prefix}_finish_reason" if key_prefix else "finish_reason"

    for sample in samples:
        if prompt_key not in sample:
            raise ValueError(
                f"Sample #{sample['sample_index']} - {sample['completion_index']} does not contain {prompt_key}")

    all_samples = samples  # keep a reference to all samples
    samples = [s for s in samples if completion_key not in s]  # filter out already completed samples for inference

    if len(samples) == 0:
        print("All {} samples have been completed.".format(len(all_samples)))
    else:
        print("Inferring completions for {} remaining samples (total={})".format(len(samples), len(all_samples)))

        pbar = tqdm(total=len(samples), desc="Inferring completions via OpenAI")
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            prompts = [s[prompt_key] for s in batch_samples]
            response = create_completion(model=model_id, prompt=prompts, max_tokens=max_tokens,
                                         temperature=temperature, n=1, stop=STOP_PHRASE)
            assert len(response["choices"]) == len(batch_samples)

            completions = [c["text"] for c in response["choices"]]
            finish_reasons = [c["finish_reason"] for c in response["choices"]]
            for sample in batch_samples:
                sample[completion_key] = completions.pop(0)
                sample[finish_reason_key] = finish_reasons.pop(0)

            if save_completion_dataset:
                save_completion_dataset.save()

            pbar.update(len(batch_samples))

    return all_samples


def populate_completion_dataset(completion_dataset: CompletionDataset, dataset: Dataset,
                                formatter: Formatter, sample_indices: List[int] = None, augs: int = 1,
                                prompt_key: str = "prompt"):
    if sample_indices is None:
        sample_indices = list(range(len(dataset.data)))

    dataset_samples = dataset.select_samples(sample_indices)
    for i, datsaet_sample in zip(sample_indices, dataset_samples):
        # Add sample lists
        if i in completion_dataset.data:
            completion_samples = completion_dataset.data.get(i)
        else:
            completion_samples = list()
            completion_dataset.data[i] = completion_samples

        # Add completion sample dicts
        remaining = augs - len(completion_samples)
        for _ in range(remaining):
            completion_samples.append(dict())

        dataset_sample = dataset.data[i]

        # Populate completion sample dicts
        for j, completion_sample in enumerate(completion_samples):
            completion_sample["sample_index"] = i
            completion_sample["completion_index"] = j
            completion_sample["question"] = dataset_sample["question"]
            completion_sample["answer"] = dataset_sample["answer"]
            if formatter.prediction_template == "zs_cot" and formatter.zs_cot_step == 2:
                if "reasoning_completion" not in completion_sample:
                    raise ValueError(
                        "All samples must contain a 'reasoning_completion' key for zs_cot step 2. Make sure to run step 1 for the same sample/completion indices")
            completion_sample[prompt_key] = formatter(completion_sample, include_label=False)["input"]


def infer_completion_data(completion_metadata: CompletionMetadata, zs_cot_step: int = None,
                          sample_indices: List[int] = None, augs: int = 1,
                          temperature: float = 0, max_tokens: int = 128):
    """
    Init/load CompletionDataset, infer completions for remaining samples, and save.

    - sample_indices: indices of samples to infer, or None for all samples
    - augs: number of completion_indices per sample
    """
    model_key = get_model_key(completion_metadata.base_model, completion_metadata.dataset_key,
                              completion_metadata.train_key)
    model_id = get_model_id(model_key)
    if model_id is None:
        raise ValueError(f"OpenAI model with model_key=`{model_key}` does not exist")

    formatter = Formatter("decoder", completion_metadata.prediction_template, zs_cot_step,
                          completion_metadata.dataset_key, stop_phrase=STOP_PHRASE)

    # If running zs_cot step 1, add "reasoning" prefix to keys
    if completion_metadata.prediction_template == "zs_cot" and zs_cot_step == 1:
        key_prefix = "reasoning"
    else:
        key_prefix = None
    prompt_key = f"{key_prefix}_prompt" if key_prefix else "prompt"
    temperature_key = f"{key_prefix}_temperature" if key_prefix else "temperature"
    max_tokens_key = f"{key_prefix}_max_tokens" if key_prefix else "max_tokens"

    if completion_metadata.prediction_template == "zs_cot" and zs_cot_step is None:
        raise ValueError("zs_cot_step must be specified for prediction_template='zs_cot'")

    # Load dataset
    dataset = Dataset.load(completion_metadata.dataset_key)
    if sample_indices is None:
        sample_indices = list(range(len(dataset.data)))

    # Load or init CompletionDataset
    try:
        completion_dataset = CompletionDataset.load(completion_metadata)
        print("Loaded {} samples from:".format(completion_dataset.total_samples))
        print(completion_dataset.path)
    except FileNotFoundError:
        completion_dataset = CompletionDataset.init(completion_metadata, additional_metadata={
            temperature_key: temperature,
            max_tokens_key: max_tokens,
        })
        print("Initializing new CompletionDataset at:")
        print(completion_dataset.path)

    # Populate CompletionDataset with formatted prompts, etc.
    populate_completion_dataset(completion_dataset, dataset, formatter, sample_indices, augs, prompt_key=prompt_key)
    completion_dataset.save()

    # Get list of individual completion sample dicts
    completion_indices = list(range(augs))
    completion_samples = completion_dataset.select_samples(sample_indices, completion_indices)

    # Infer completions
    batch_infer_samples(completion_samples, model_id, key_prefix, batch_size=20, temperature=temperature,
                        max_tokens=max_tokens, save_completion_dataset=completion_dataset)

    return completion_dataset
