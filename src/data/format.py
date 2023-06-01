"""
Format Dataset or CompletionDataset samples for (CoT) inference and fine-tuning.
"""
import copy
import sys
from typing import Dict, List

from data.few_shot_cot_prompt import get_few_shot_cot_prompt

SUPPORTED_MODEL_TYPES = ["decoder", "encoder_decoder"]
SUPPORTED_PREDICTION_TEMPLATES = ["ft_token", "ft_cot_natural", "ft_cot_token", "zs", "zs_cot", "fs_cot"]


class Formatter:
    def __init__(self, model_type: str, prediction_template: str, zs_cot_step: int = None, dataset_key: str = None,
                 stop_phrase: str = None):
        """
        Parameters

        - model_type
        - prediction_template
        - zs_cot_step
        - dataset_key
        - stop_phrase: used as `stop` string for OpenAI API. YOU MUST SET THIS VALUE FOR OPENAI FINE-TUNING or the model
          will not learn to stop. our experiments.
        """
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise NotImplementedError("model_type={}".format(model_type))
        if prediction_template not in SUPPORTED_PREDICTION_TEMPLATES:
            raise NotImplementedError("prediction_template={}".format(prediction_template))

        self.model_type = model_type
        self.prediction_template = prediction_template
        self.zs_cot_step = zs_cot_step
        self.dataset_key = dataset_key
        self.stop_phrase = stop_phrase  # used as `stop` string for OpenAI API

        if prediction_template == "ft_natural":
            if model_type == "decoder":
                self.input_format = "Q: {sample[question]}\n\nA:"
                self.label_format = " {sample[answer]}"
                # REPRODUCTION NOTE - GPT3 experiments in the paper use the following:
                # self.input_format = "{sample[question]}\n\n ### \n\n"
                # self.label_format = " {sample[answer]}"
            elif model_type == "encoder_decoder":
                self.input_format = "Q: {sample[question]}"
                self.label_format = "{sample[answer]}"

        if prediction_template == "ft_token":
            if model_type == "decoder":
                self.input_format = "{sample[question]} ###"
                self.label_format = " {sample[answer]}"
                # REPRODUCTION NOTE - GPT3 experiments in the paper use the following:
                # self.input_format = "{sample[question]}\n\n ### \n\n"
                # self.label_format = " {sample[answer]}"
            elif model_type == "encoder_decoder":
                self.input_format = "{sample[question]}"
                self.label_format = "{sample[answer]}"

        if prediction_template == "ft_cot_token":
            if model_type == "decoder":
                self.input_format = "{sample[question]} ###"
                self.label_format = " {sample[reasoning_completion]} --> {sample[answer]}"
                # REPRODUCTION NOTE - GPT3 experiments in the paper use the following:
                # self.input_format = "{sample[question]}\n\n###\n\n"
                # self.label_format = " {sample[reasoning_completion]}\n\n-->\n\n{sample[answer]}"
            elif model_type == "encoder_decoder":
                self.input_format = "{sample[question]}"
                self.label_format = "{sample[reasoning_completion]} --> {sample[answer]}"

        if prediction_template == "ft_cot_natural":
            if model_type == "decoder":
                self.input_format = "Q: {sample[question]}\n\nA: Let's think step by step.\n\n"
                self.label_format = " {sample[reasoning_completion]}\n\nTherefore the answer is {sample[answer]}"
                # REPRODUCTION NOTE - GPT3 experiments in the paper use the following:
                # self.input_format = "Q: {sample[question]}\n\nA: Let's think step by step.\n\n"
                # self.label_format = " {sample[reasoning_completion]}\n\nTherefore the answer is\n\n{sample[answer]}"
            else:
                raise NotImplementedError("model_type={} not supported for prediction_template={}".format(
                    model_type, prediction_template))

        if prediction_template == "zs":
            self.label_format = None
            if model_type == "decoder":
                self.input_format = "Q: {sample[question]}\n\nA:"
            elif model_type == "encoder_decoder":  # following SQuAD format used to train T5
                self.input_format = "question: {sample[question]}"
            else:
                raise NotImplementedError("model_type={} not supported for zs_cot".format(model_type))

        if prediction_template == "zs_cot":
            self.label_format = None
            if model_type == "decoder":
                if zs_cot_step == 1:
                    self.input_format = "Q: {sample[question]}\nA: Let's think step by step."
                elif zs_cot_step == 2:
                    self.input_format = "{sample[reasoning_prompt]}{sample[reasoning_completion]}\nTherefore, the answer is"
                else:
                    raise ValueError("step {} not supported for zs_cot".format(zs_cot_step))
            else:
                raise NotImplementedError("model_type={} not supported for zs_cot".format(model_type))

        if prediction_template == "fs_cot":
            self.label_format = None
            if dataset_key is None:
                raise ValueError("dataset_key must be specified for fs_cot")
            self.few_shot_prompt = get_few_shot_cot_prompt(dataset_key)

            if model_type == "decoder":
                self.input_format = self.few_shot_prompt + "\nQ: {sample[question]}\nA:"
            elif model_type == "encoder_decoder":
                self.input_format = self.few_shot_prompt + "\nQ: {sample[question]}\nA:"
            else:
                raise NotImplementedError("model_type={} not supported for fs_cot".format(model_type))

        if not hasattr(self, "input_format"):
            raise NotImplementedError(f"{model_type}, {prediction_template}")
        if not hasattr(self, "label_format"):
            # should be set to None if not supported
            raise NotImplementedError(f"{model_type}, {prediction_template}")

    def __call__(self, sample: Dict, include_label: bool = False):
        """
        Sample can either be a dataset sample (from Dataset) or completion sample (from CompletionDataset).
        Samples should contain all necessary keys needed for self.prediction_template, e.g., zs_cot step 2 requires
        reasoning_completion.
        """
        sample = copy.deepcopy(sample)

        # REPRODUCTION NOTE - stripping is not applied to GPT3 experiments in the paper.
        if "question" in sample:
            sample["question"] = sample["question"].strip()
        if "answer" in sample:
            sample["answer"] = sample["answer"].strip()
        if "reasoning_completion" in sample:
            sample["reasoning_completion"] = sample["reasoning_completion"].strip()
        if "reasoning_prompt" in sample:
            sample["reasoning_prompt"] = sample["reasoning_prompt"].strip()

        result = {
            "sample_index": sample["sample_index"],
            "input": self.input_format.format(sample=sample),
        }

        if include_label:
            if self.label_format is None:
                raise ValueError("label formatting is not supported for prediction_template={}".format(
                    self.prediction_template))
            result["label"] = self.label_format.format(sample=sample)

            if self.stop_phrase is not None:
                result["label"] += " " + self.stop_phrase

        return result

    def format_samples(self, samples: List[Dict], include_labels: bool = False) -> List[Dict]:
        """
        - samples: List of samples from Dataset or CompletionDataset. Use the `select_samples()` method
          provided by either class.
        """
        formatted_samples = []

        errors = 0
        for sample in samples:
            try:
                formatted_sample = self(sample, include_label=include_labels)
            except ValueError as e:
                errors += 1
                continue
            formatted_samples.append(formatted_sample)

        if errors > 0:
            print("ERROR: {}/{} samples could not be formatted".format(errors, len(samples)), file=sys.stderr)
            print("Raising last Exception", file=sys.stderr)
            raise e

        return formatted_samples
