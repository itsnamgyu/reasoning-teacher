import re
from collections import defaultdict

import pandas as pd

ANSWER_PREFIXES = {
    None: None,
    "natural": "Therefore, the answer is",
    "special": "-->",
    "few_shot_cot": "The answer is",
}


def extract_predictions(prediction, dataset_key):
    if dataset_key in ("aqua", "commonsense_qa"):
        prediction = re.findall(r'A|B|C|D|E', prediction)
    elif dataset_key == "date_understanding":
        prediction = re.findall(r'A|B|C|D|E|F', prediction)
    elif dataset_key in ("tracking_shuffled_objects"):
        prediction = re.findall(r'A|B|C', prediction)
    elif dataset_key in ("gsm8k", "addsub", "multiarith", "svamp", "single_eq"):
        prediction = prediction.replace(",", "")
        prediction = [s for s in re.findall(r'-?\d+(?:\.\d+)?', prediction)]
        if dataset_key in ("addsub", "svamp", "single_eq"):
            prediction = [float(s) for s in prediction]
    elif dataset_key in ("strategy_qa", "coin_flip"):
        prediction = prediction.lower()
        prediction = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", prediction)
        prediction = prediction.split(" ")
        prediction = [i for i in prediction if i in ("yes", "no")]
    elif dataset_key == "last_letter_concatenation":
        prediction = re.sub("\"|\'|\n|\.|\s", "", prediction)
        prediction = [prediction]
    else:
        raise ValueError("Invalid dataset: {}".format(dataset_key))

    return prediction


def cleanse_prediction(completion, dataset_key, answer_prefix=None, return_all=True):
    if answer_prefix is not None:
        # Use first candidate after prefix if found, else last candidate
        index = completion.find(answer_prefix)
        if index == -1:
            answers = extract_predictions(completion, dataset_key)
            first = False
        else:
            start_of_answer = index + len(answer_prefix)
            answers = extract_predictions(completion[start_of_answer:], dataset_key)
            first = True
    else:
        # If no prefix, use first candidate
        answers = extract_predictions(completion, dataset_key)
        first = True

    answer = None
    if answers:
        answer = (answers[0] if first else answers[-1])

    return (answer, answers) if return_all else answer


def cleanse_answer(answer: str, dataset_key):
    if dataset_key in ("gsm8k", "addsub", "multiarith", "svamp", "single_eq"):
        answer = answer.replace(",", "")
    if dataset_key == "strategy_qa":
        answer = answer.lower()
    if dataset_key in ("addsub", "svamp", "single_eq"):
        answer = float(answer)

    return answer


def compare_prediction_and_answer(prediction, answer, dataset_key) -> bool:
    if dataset_key in ("addsub", "svamp", "single_eq"):
        return prediction is not None and abs(prediction - answer) <= 1e-6
    else:
        return prediction is not None and prediction == answer


DEFAULT_AUGMENTATIONS = 1


def evaluate_completions(completion_data, dataset_key, template="special", print_metrics=False,
                         indices=None, augmentations=DEFAULT_AUGMENTATIONS, augmentation_indices=None):
    if indices is None:
        indices = list(completion_data.keys())
    answer_prefix = ANSWER_PREFIXES[template]

    if augmentation_indices is None:
        augmentation_indices = list(range(augmentations))
    else:
        if augmentations != DEFAULT_AUGMENTATIONS:
            raise ValueError(
                "You should not specify `augmentations` when you have explicitly specified `augmentation_indices`")

    evaluation_data = defaultdict(list)
    for index in indices:
        samples = completion_data[index]
        selected_samples = []
        for i in augmentation_indices:
            try:
                selected_samples.append(samples[i])
            except IndexError:
                # Ensure that all indices have all augmentations (i.e., total samples = num indices * num augmentations)
                error = "Augmentations #{} does not exist for sample #{}".format(i, index)
                raise Exception(error)
        for s in selected_samples:
            completion = s["completion"]
            if answer_prefix is None:
                contains_answer_prefix = None
            else:
                contains_answer_prefix = answer_prefix in completion
            prediction, candidates = cleanse_prediction(completion, dataset_key, answer_prefix=answer_prefix)
            answer = s["answer"]
            answer = cleanse_answer(answer, dataset_key)

            correct = compare_prediction_and_answer(prediction, answer, dataset_key)
            contains_answer = False
            for p in candidates:
                if compare_prediction_and_answer(p, answer, dataset_key):
                    contains_answer = True

            evaluation_data["sample_index"].append(index)
            evaluation_data["contains_prediction"].append(prediction is not None)
            evaluation_data["correct"].append(correct)
            evaluation_data["contains_answer"].append(contains_answer)
            evaluation_data["contains_answer_prefix"].append(contains_answer_prefix)

            if "reasoning_finish_reason" in s:
                evaluation_data["reasoning_finish_reason"].append(s["reasoning_finish_reason"])
            if "finish_reason" in s:
                evaluation_data["finish_reason"].append(s["finish_reason"])
                complete = s.get("finish_reason", None) == "stop"
                evaluation_data["complete"].append(complete)

    evaluation = pd.DataFrame(evaluation_data)

    if print_metrics:
        metrics = get_evaluation_metrics(evaluation)
        for key, value in metrics.items():
            print("{:40s}: {:7.3f}".format(key, value * 100))

    return pd.DataFrame(evaluation_data)


def get_evaluation_metrics(evaluation: pd.DataFrame):
    correct = evaluation.correct
    contains_prefix = evaluation.contains_answer_prefix
    contains_prediction = evaluation.contains_prediction
    contains_answer = evaluation.contains_answer  # among all prediction candidates in the completion
    metrics = {
        "total": len(evaluation),
        "accuracy": correct.sum() / len(evaluation),
        "correct": correct.sum(),
        "contains_prediction": contains_prediction.sum() / len(evaluation),
        "contains_answer": contains_answer.sum() / len(evaluation),
    }

    # Note, all samples *should* have the same number of augmentations for every sample
    augmentations = evaluation.groupby("sample_index").size().max()
    if augmentations > 1:
        at_least_one_correct = evaluation.groupby("sample_index").correct.sum() > 0
        samples = len(evaluation.groupby("sample_index"))
        metrics.update({
            "samplewise_accuracy": at_least_one_correct.sum() / samples,
            "samplewise_correct": at_least_one_correct.sum(),
        })

    assert contains_prefix.isnull().any() == contains_prefix.isnull().all()
    if not contains_prefix.isnull().any():
        metrics.update({
            "contains_prefix": contains_prefix.sum() / len(evaluation),
            "accuracy_with_prefix": (contains_prefix & correct).sum() / contains_prefix.sum(),
            "accuracy_without_prefix": (~contains_prefix & correct).sum() / (~contains_prefix).sum(),
        })
    if "reasoning_finish_reason" in evaluation.columns:
        complete = evaluation.reasoning_finish_reason.eq("stop")
        metrics.update({
            "reason_complete": complete.sum() / len(evaluation),
            "accuracy_when_reason_complete": (complete & correct).sum() / complete.sum(),
            "accuracy_when_reason_incomplete": (~complete & correct).sum() / (~complete).sum(),
        })

    if "complete" in evaluation.columns:
        complete = evaluation.complete
        metrics.update({
            "complete": complete.sum() / len(evaluation),
            "accuracy_when_complete": (complete & correct).sum() / complete.sum(),
            "accuracy_when_incomplete": (~complete & correct).sum() / (~complete).sum(),
        })

    return metrics
