# Large Language Models Are Reasoning Teachers


Official repository for [Large Language Models Are Reasoning Teachers](https://arxiv.org/abs/2212.10071), by
Namgyu Ho, Laura Schmid, and Se-young Yun.

**🚀 Accepted to ACL 2023.**

This repository contains code for (1) running CoT reasoning on OpenAI models,
and (2) apply Fine-tune-CoT to train students based on OpenAI models *or* custom models such as T5, GPT-2 on your GPUs.

## Getting Started

### OpenAI API Experiments

OpenAI API experiments are implemented in the `oai` module. Refer to `notebooks/example_oai_finetune_cot.ipynb`
on how to run Fine-tune-CoT from start to finish.

### Custom Experiments (on GPU) 

Custom experiments are implemented in the `custom` module, based on PyTorch Lightning. Refer to `main.py`
and `scripts/custom/*.sh` on how to fine-tune models such as T5, Flan-T5, and GPT-2 using Fine-tune-CoT.



## Setup

```
pip install -r requirements.txt
python setup.py develop
```

## Resources

### Teacher, student inference data

Will be available for download soon!

### Template-based split (paper Appendix E.3)

Template-based splits for MultiArith and Date Understanding are saved in `/data/splits/*__template.json`

### Few-shot Prompts

Few-shot prompts adapted from Wei 2022 are saved in `/data/few_shot_cot_prompts.json`

## Data Structures

### `data.dataset.Dataset`

```json
{
  "metadata": {
    "dataset_key": "multiarith"
  },
  "data": [
    {
      "sample_index": 0,
      "question": "string",
      "answer": "string",
      "rationale": "string?"
    }
  ]
}
```

### `data.completion.CompletionDataset`

```json
{
  "metadata": {
    "dataset_key": "multiarith",
    "base_model": "curie",
    "finetune_key": "zs_cot_multiarith",
    "train_key": "ft_cot",
    "prediction_template": "ft_cot_token",
  },
  "data": {
    "<sample_index>": [
      {
        "sample_index": 0,
        "completion_index": 0,
        "question": "string",
        "answer": "string",
        "prompt": "string",
        "completion": "string",
        "finish_reason": "string",
        "reasoning_prompt": "string?",
        "reasoning_completion": "string?",
        "reasoning_finish_reason": "string?",
      }
    ]
  }
}
```

## Data Organization

*Needs update.*

- `<model_key>` = `B_<base_model>_T_<train_key>`

### File Organization Pattern

```
saved/
|–– completion_data/
    |–– B_<BASE_MODEL>__C_<COMPLETION_KEY>/
        |-- D_<DATESET_KEY>.json  # base model inference
        |-- F_<FINETUNE_KEY>__D_<DATESET_KEY>.json  # default fine-tuned model inference
        |-- F_<FINETUNE_KEY>__T_<TRAIN_KEY>__D_<DATESET_KEY>.json  # custom fine-tuned model inference
|–– finetune_data/
    |–– P_<PLATFORM_KEY>/
        |–– F_<FINETUNE_KEY>{.*|/}
|–– model_metadata/
    |–– B_<base_model>
        |–– F_<FINETUNE_KEY>__T_<train_key>.json
```

### File Organization Examples

```
saved/
|–– completion_data/
    |–– B_text-davinci-002__C_zs_cot/
    |–– B_text-davinci-002__C_zs_cot_long/
    |–– B_text-davinci-002__C_fs_cot/
    |–– B_curie__C_zs_cot/
    |–– B_curie__C_fs_cot/
    |–– B_curie__C_zs/
    |–– B_curie__C_ft_cot/
|–– finetune_data/
    |–– F_zs_cot_multiarith/  # text-davinci-002_zs_cot
    |–– F_zs_cot_long_multiarith/
|–– model_metadata/
    |–– B_curie/
        |–– F_zs_cot_multiarith.json
```


### Personal Note

![accepted](acl2023.jpg)

