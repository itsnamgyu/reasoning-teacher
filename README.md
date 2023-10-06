# Large Language Models Are Reasoning Teachers


Official repository for [Large Language Models Are Reasoning Teachers](https://arxiv.org/abs/2212.10071), by
Namgyu Ho, Laura Schmid, and Se-young Yun.

**🚀 Accepted to ACL 2023.**

This repository contains code for (1) running CoT reasoning on OpenAI models,
and (2) apply Fine-tune-CoT to train students based on OpenAI models *or* custom open-source models such as T5, Flan-T5, GPT-2 on your GPUs, based on 🤗 and Pytorch Lightning.


## Getting Started

### OpenAI API Experiments

OpenAI API experiments are implemented in the `oai` module. Refer to `notebooks/example_oai_finetune_cot.ipynb`
on how to run Fine-tune-CoT from start to finish.

### Custom Experiments (on GPU) 

Custom experiments are implemented in the `custom` module, based on PyTorch Lightning. Refer to `custom_train.py`
and `scripts/custom/*.sh` on how to fine-tune models such as T5, Flan-T5, and GPT-2 using Fine-tune-CoT.

## Setup

```
pip install -r requirements.txt
python setup.py develop
```

### Environment

The code has been tested on Python<=3.10, PyTorch Lightning<=1.9, PyTorch>=2.0

## Data 🚀

We're proud to share *all* of our raw experimental data! All data is organized in json or jsonl format, for your pleasure :)

Cloud storage folder links:

- [Dropbox](https://www.dropbox.com/sh/hwcncpyomx87h20/AACqgVdd-ZzBQ3ncJcKqw0cVa?dl=0)
- [Google Drive](https://drive.google.com/drive/folders/1C6kah3WV36N8omlUl-TeU9tsJADZNaJV?usp=share_link)

### File List

- `dataset.tar.gz`: 12 task datasets compiled in a unified json format
  - Belongs in `PROJECT/data/dataset/`
- `completion_data.tar.gz`: Completion data, i.e., inference data, from all teachers and students, for *all* experiments. About 8GB when uncompressed
  - Belongs in `PROJECT/saved/completion_data/`
- `teacher_completion_data.tar.gz`: Completion data from Zero-shot-CoT (with diverse reasoning) on the default teacher model `text-davinci-002` using the OpenAI API. About 💰 $1000+ worth of goods, with ❤️ from [OSI LAB](http://osi.kaist.ac.kr) at [KAIST](https://kaist.ac.kr) . Subset of `completion_data.tar.gz`.
  - Belongs in `PROJECT/saved/completion_data/`.
- `finetune_data.tar.gz`: *All* data used to fine-tune OpenAI students via the fine-tuning API, in jsonl format. These are derived from teacher completion data and can be generated from our code.
  - Belongs in `PROJECT/saved/finetune_data/`

### Generate Paper Results

After downloading the full `completion_data.tar.gz`, you can run `notebooks/results.ipynb` to generate *all* result tables and figures from our paper. The code will (re-)evaluate all raw text model outputs contained in the completion data.



## Additional Resources

### Template-based Split (Paper Appendix E.3)

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

