import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, "dataset")
SPLIT_PATH = os.path.join(DATA_PATH, "split")
FEW_SHOT_COT_PROMPTS_PATH = os.path.join(DATA_PATH, "few_shot_cot_prompts.json")

SAVED_PATH = os.path.join(PROJECT_PATH, "saved")
FINETUNE_DATA_PATH = os.path.join(SAVED_PATH, "finetune_data")
COMPLETION_DATA_PATH = os.path.join(SAVED_PATH, "completion_data")


def get_dataset_path(dataset_key: str) -> str:
    return os.path.join(DATASET_PATH, "{}.json".format(dataset_key))


def get_split_path(dataset_key: str, split_key: str) -> str:
    return os.path.join(SPLIT_PATH, "{}__{}.json".format(dataset_key, split_key))


def get_completion_data_path(base_model: str, completion_key: str, dataset_key: str,
                             train_key: str = None, epoch: int = None) -> str:
    dirname = "B_{}__C_{}".format(base_model, completion_key)
    base_tags = ["D_{}".format(dataset_key)]
    if train_key is not None:
        base_tags.append("T_{}".format(train_key))
    if epoch is not None:
        base_tags.append("E_{:03d}".format(epoch))
    basename = "__".join(base_tags)

    return os.path.join(COMPLETION_DATA_PATH, dirname, "{}.json".format(basename))


def get_finetune_data_path(platform_key: str, finetune_key: str) -> str:
    if platform_key == "openai":
        return os.path.join(FINETUNE_DATA_PATH, "P_{}".format(platform_key), "F_{}.jsonl".format(finetune_key))
    else:
        return os.path.join(FINETUNE_DATA_PATH, "P_{}".format(platform_key), "F_{}.json".format(finetune_key))
