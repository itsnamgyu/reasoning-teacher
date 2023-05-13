import warnings

import pandas as pd


def summarize_evaluation(evaluation: pd.DataFrame) -> dict:
    """
    Summarize metrics from completion-wise evaluation dataframe.
    Dataframe contains columns "sample_index", "completion_index", "correct", "contains_answer", "correct_format", "complete",
    """
    if evaluation is None or len(evaluation) == 0:
        warnings.warn("No completions to evaluate.")
        return None

    return {
        "accuracy": evaluation.correct.mean(),
        "contains_answer": evaluation.contains_answer.mean(),
        "correct_format": evaluation.correct_format.mean(),
        "complete": evaluation.complete.mean(),
    }

