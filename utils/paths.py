"""
Paths used commonly throughout the project.
Paths pertaining to particular modules are defined there.
"""
import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")
CUSTOM_SPLIT_PATH = os.path.join(DATASET_PATH, "custom_split")
SAVED_PATH = os.path.join(PROJECT_PATH, "saved")
FINETUNE_DATA_PATH = os.path.join(SAVED_PATH, "finetune_data")
COMPLETION_DATA_PATH = os.path.join(SAVED_PATH, "completion_data")
FEW_SHOT_COT_PROMPTS_PATH = os.path.join(DATASET_PATH, "few_shot_cot_prompts.json")