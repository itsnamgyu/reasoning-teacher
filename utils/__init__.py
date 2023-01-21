import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATASET_PATHS = {
    "aqua": "./dataset/AQuA/test.json",
    "gsm8k": "./dataset/grade-school-math/test.jsonl",
    "commonsensqa": "./dataset/CommonsenseQA/dev_rand_split.json",
    "addsub": "./dataset/AddSub/AddSub.json",
    "multiarith": "./dataset/MultiArith/MultiArith.json",
    "strategyqa": "./dataset/StrategyQA/task.json",
    "svamp": "./dataset/SVAMP/SVAMP.json",
    "singleeq": "./dataset/SingleEq/questions.json",
    "bigbench_date": "./dataset/Bigbench_Date/task.json",
    "object_tracking": "./dataset/Bigbench_object_tracking/task.json",
    "coin_flip": "./dataset/coin_flip/coin_flip.json",
    "last_letters": "./dataset/last_letters/last_letters.json"
}
