import re
from typing import List, Tuple, Union, Optional, Dict

import pandas as pd

from data.completion_dataset import CompletionDataset

PREDICTION_PREFIXES = {
    None: None,
    "zs": None,
    "ft_natural": None,
    "ft_token": None,
    "fs_cot": "The answer is",
    "zs_cot": None,
    "ft_cot_natural": "Therefore, the answer is",
    "ft_cot_token": "-->",
}


class Evaluator:
    dataset_key: str
    prediction_template: Optional[str]
    prediction_prefix: Optional[str]

    def __init__(self, dataset_key: str, prediction_template: str):
        """
        Set prediction_template=None if you are only using the evaluator to parse answers.
        """
        self.dataset_key = dataset_key
        self.prediction_template = prediction_template
        if prediction_template not in PREDICTION_PREFIXES:
            raise ValueError("Invalid prediction template: {}".format(prediction_template))
        else:
            self.prediction_prefix = PREDICTION_PREFIXES[prediction_template]

    @staticmethod
    def for_completion_dataset(completion_dataset: CompletionDataset) -> "Evaluator":
        return Evaluator(completion_dataset.dataset_key, completion_dataset.prediction_template)

    @staticmethod
    def evaluate_completion_dataset(completion_dataset: CompletionDataset, sample_indices: List[int] = None,
                                    completion_indices: List[int] = None) -> pd.DataFrame:
        """
        Evaluate a set of completions (i.e. a CompletionData object).

        - indices: If not None, only evaluate completions for the given sample indices.
        - completion_indices: If not None, only evaluate the completions with the specified indices for each sample,
          e.g., to evaluate repeated completions with temperature sampling.
        """
        evaluator = Evaluator.for_completion_dataset(completion_dataset)
        completions = completion_dataset.select_samples(sample_indices, completion_indices)
        evaluations = []
        for completion in completions:
            evaluations.append(evaluator.evaluate_completion(completion))

        return pd.DataFrame(evaluations)

    def evaluate_completion(self, completion: Dict) -> Dict:
        """
        Evaluate a single prediction.
        """
        completion_string = completion["completion"]
        correct_format = self.prediction_prefix is None or completion_string.find(self.prediction_prefix) != -1
        prediction, candidates = self.cleanse_prediction(completion_string, return_all=True)
        answer = self.cleanse_answer(completion["answer"])
        return {
            "sample_index": completion["sample_index"],
            "completion_index": completion["completion_index"],
            "correct": self._compare_prediction_and_answer(prediction, answer),
            "contains_answer": any(self._compare_prediction_and_answer(p, answer) for p in candidates),
            "correct_format": correct_format,
            "complete": completion.get("finish_reason") == "stop",
        }

    def check_answer(self, completion_string: str, answer: str) -> bool:
        """
        Check if a single prediction is correct.
        """
        prediction = self.cleanse_prediction(completion_string, return_all=False)
        answer = self.cleanse_answer(answer)
        return self._compare_prediction_and_answer(prediction, answer)

    def cleanse_prediction(self, completion: str, return_all: bool) -> Union[str, Tuple[str, List[str]]]:
        if self.prediction_prefix is None:
            # If no prefix, use first candidate
            predictions = self._extract_prediction_candidates(completion)
            first = True
        else:
            index = completion.find(self.prediction_prefix)
            if index == -1:
                # If prefix not found, use *last* candidate
                predictions = self._extract_prediction_candidates(completion)
                first = False
            else:
                # If prefix found, use *first* candidate after prefix
                start_of_answer = index + len(self.prediction_prefix)
                predictions = self._extract_prediction_candidates(completion[start_of_answer:])
                first = True

        answer = None
        if predictions:
            answer = (predictions[0] if first else predictions[-1])

        return (answer, predictions) if return_all else answer

    def cleanse_answer(self, answer: str) -> str:
        if self.dataset_key in ["gsm8k", "addsub", "multiarith", "svamp", "single_eq"]:
            answer = answer.replace(",", "")
        if self.dataset_key == "strategy_qa":
            answer = answer.lower()
        if self.dataset_key in ["addsub", "svamp", "single_eq"]:
            answer = float(answer)

        return answer

    def _extract_prediction_candidates(self, prediction: str) -> List[str]:
        """
        Extracts all potential answer predictions which satisfy the dataset's answer format from the
        prediction string
        """
        if self.dataset_key in ("aqua", "commonsense_qa"):
            prediction = re.findall(r'[ABCDE]', prediction)
        elif self.dataset_key == "date_understanding":
            prediction = re.findall(r'[ABCDEF]', prediction)
        elif self.dataset_key in ("tracking_shuffled_objects"):
            prediction = re.findall(r'[ABC]', prediction)
        elif self.dataset_key in ("gsm8k", "addsub", "multiarith", "svamp", "single_eq"):
            prediction = prediction.replace(",", "")
            prediction = re.findall(r'-?\d+(?:\.\d+)?', prediction)
            if self.dataset_key in ("addsub", "svamp", "single_eq"):
                prediction = [float(s) for s in prediction]
        elif self.dataset_key in ("strategy_qa", "coin_flip"):
            prediction = prediction.lower()
            prediction = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", prediction)
            prediction = prediction.split(" ")
            prediction = [i for i in prediction if i in ("yes", "no")]
        elif self.dataset_key == "last_letter_concatenation":
            prediction = re.sub("\"|\'|\n|\.|\s", "", prediction)
            prediction = [prediction]
        else:
            raise ValueError("Invalid dataset: {}".format(self.dataset_key))

        return prediction

    def _compare_prediction_and_answer(self, prediction, answer) -> bool:
        if self.dataset_key in ("addsub", "svamp", "single_eq"):
            return prediction is not None and abs(prediction - answer) <= 1e-6
        else:
            return prediction is not None and prediction == answer
