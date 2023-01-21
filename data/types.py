from typing import TypedDict, Optional


class DatasetSample(TypedDict):
    question: str
    answer: str
    reasoning: Optional[str]


class CompletionSample(TypedDict):
    sample_index: int
    question: str
    answer: str
    reasoning_prompt: Optional[str]
    reasoning_completion: Optional[str]
    reasoning_finish_reason: Optional[str]
    prompt: str
    completion: str
    finish_reason: Optional[str]
