import json
from typing import List

from paths import FEW_SHOT_COT_PROMPTS_PATH


def load_few_shot_cot_prompts():
    with open(FEW_SHOT_COT_PROMPTS_PATH) as f:
        return json.load(f)


def get_few_shot_cot_prompt(dataset_key) -> str:
    data = load_few_shot_cot_prompts()
    if dataset_key not in data:
        raise KeyError("Few-shot-CoT prompts are not available for dataset `{}`".format(dataset_key))
    return data[dataset_key]["prompt"]


def get_few_shot_cot_sample_indices(dataset_key) -> List[int]:
    data = load_few_shot_cot_prompts()
    if dataset_key not in data:
        raise KeyError("Few-shot-CoT prompts are not available for dataset `{}`".format(dataset_key))
    return data[dataset_key]["sample_indices"]
