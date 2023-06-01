import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from custom.utils import save_finetune_data, load_finetune_data, list_of_dicts_to_dict_of_lists
from data.completion_dataset import CompletionDataset, CompletionIdentifier
from data.dataset import DATASET_KEYS, Dataset
from data.format import Formatter
from data.split import load_train_test_split

SUPPORTED_KEYS = ["zs", "fs_cot", "ft", "ft_cot"]
SUPPORTED_KEYS += ["ft_cot_t70_{}aug".format(aug) for aug in [1, 2, 4, 8, 16, 32, 64]]
SUPPORTED_MODEL_TYPES = ["decoder", "encoder_decoder"]


class DataModule(pl.LightningDataModule):
    train_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, dataset_key: str, preset_key: str, tokenizer, model_type: str, batch_size: int = 32,
                 inference_batch_size=None, num_workers: int = 8, append_eos=False):
        """
        Note that padding is applied manually on the left side when `model_type=decoder`, therefore
        `tokenizer.padding_side` is irrelevant.

        - model_type: `decoder` or `encoder_decoder`. Used as `platform_key` for saving fine-tune data.
        - append_eos: manually append eos token to the end of the label.
        """
        super().__init__()
        if dataset_key not in DATASET_KEYS:
            raise NotImplementedError("dataset_key={}".format(dataset_key))
        if preset_key not in SUPPORTED_KEYS:
            raise NotImplementedError("Not implemented: key={}".format(preset_key))
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise NotImplementedError("Not implemented: model_type={}".format(model_type))

        self.dataset_key = dataset_key
        self.preset_key = preset_key
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.append_eos = append_eos

        if self.preset_key == "zs":
            self.finetune_key = None
            self.prediction_template = "zs"
        if self.preset_key == "zs_cot":
            self.finetune_key = None
            self.prediction_template = "zs_cot"
        if self.preset_key == "fs_cot":
            self.finetune_key = None
            self.prediction_template = "fs_cot"
        if self.preset_key == "ft":
            self.finetune_key = self.dataset_key
            self.prediction_template = "ft_token"
        if self.preset_key == "ft_cot":
            self.finetune_key = "zs_cot_{}".format(self.dataset_key)
            self.prediction_template = "ft_cot_token"
        for aug in [1, 2, 4, 8, 16, 32, 64]:
            if self.preset_key == "ft_cot_t70_{}aug".format(aug):
                self.finetune_key = "zs_cot_t70_{}_{}aug".format(self.dataset_key, aug)
                self.prediction_template = "ft_cot_token"

    def save_finetune_data(self, train_data):
        save_finetune_data(train_data, platform_key=self.model_type, finetune_key=self.finetune_key)

    def load_finetune_data(self):
        return load_finetune_data(platform_key=self.model_type, finetune_key=self.finetune_key)

    def setup(self, stage: str = None):
        """
        Compile train/test data. Train data (i.e., finetune_data) is saved to disk according to finetune_key.
        """
        if stage == "fit":
            train_data = self.load_finetune_data()
            if train_data is None:
                train_data = self._compile_data(train=True)
                self.save_finetune_data(train_data)
            dataset = datasets.Dataset.from_dict(train_data)
            dataset = dataset.map(self.tokenize, batched=True, batch_size=len(dataset))
            if self.model_type == "decoder":
                dataset.set_format(type="torch", columns=["sample_index", "input_ids", "attention_mask", "labels"])
            elif self.model_type == "encoder_decoder":
                dataset.set_format(type="torch",
                                   columns=["sample_index", "input_ids", "attention_mask", "decoder_attention_mask",
                                            "labels"])
            else:
                raise NotImplementedError(self.model_type)
            self.train_dataset = dataset

        test_data = self._compile_data(train=False)
        dataset = datasets.Dataset.from_dict(test_data)
        dataset = dataset.map(self.tokenize, batched=True, batch_size=len(dataset))
        dataset.set_format(type="torch", columns=["sample_index", "input_ids", "attention_mask"])
        self.test_dataset = dataset

    def tokenize(self, example) -> dict:
        if self.append_eos and "label" in example:
            for i in range(len(example["label"])):
                example["label"][i] += self.tokenizer.eos_token

        if self.preset_key == "fs_cot":
            input_max_length = 768
        else:
            input_max_length = 512

        if self.model_type == "encoder_decoder":
            it = self.tokenizer(
                example["input"],
                padding="longest",
                max_length=input_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = it["input_ids"]
            attention_mask = it["attention_mask"]
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            if "label" in example:
                lt = self.tokenizer(
                    example["label"],
                    padding="longest",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                label_ids = lt["input_ids"]
                decoder_attention_mask = lt["attention_mask"]
                label_ids[~decoder_attention_mask.bool()] = -100
                result.update({
                    "decoder_attention_mask": decoder_attention_mask,
                    "labels": label_ids,
                })
            return result
        elif self.model_type == "decoder":
            # Tokenize and apply left side padding manually

            # Tokenize in vanilla Python list form
            it = self.tokenizer(
                example["input"],
                max_length=input_max_length,
                truncation=True
            )
            iids = it["input_ids"]
            if "label" in example:
                lids = self.tokenizer(
                    example["label"],
                    max_length=512,
                    truncation=True
                )["input_ids"]
            else:
                lids = [list() for _ in range(len(iids))]

            lengths = []
            input_ids = []
            attention_mask = []
            label_ids = []
            for iid, lid in zip(iids, lids):
                lengths.append(len(iid) + len(lid))
                input_ids.append(iid + lid)
                attention_mask.append([1] * (len(iid) + len(lid)))
                label_ids.append([-100] * len(iid) + lid)

            # Pad full sequences
            lengths = torch.tensor(lengths)
            pad_lengths = (lengths.max() - lengths).tolist()
            for i, l in enumerate(pad_lengths):
                # Apply left side padding
                # Why? https://github.com/huggingface/transformers/issues/3021#issuecomment-1231526631
                input_ids[i] = [self.tokenizer.pad_token_id] * l + input_ids[i]
                attention_mask[i] = [0] * l + attention_mask[i]
                label_ids[i] = [-100] * l + label_ids[i]
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            }
        else:
            raise NotImplementedError(self.model_type)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def _compile_data(self, train=False):
        train_indices, test_indices = load_train_test_split(self.dataset_key)
        sample_indices = train_indices if train else test_indices

        if self.preset_key == "fs_cot" and train:
            raise NotImplementedError("fs_cot is not implemented for training.")

        # Get list of samples based on preset key.
        samples = None
        if self.preset_key in ["zs", "fs_cot", "ft"] or not train:
            dataset = Dataset.load(self.dataset_key)
            samples = dataset.select_samples(sample_indices)
        elif "ft_cot" in self.preset_key:
            if self.preset_key == "ft_cot":
                completion_identifier = CompletionIdentifier("text-davinci-002", "zs_cot", self.dataset_key)
                completion_indices = [0]
            else:
                for aug in [1, 2, 4, 8, 16, 32, 64]:
                    if self.preset_key == "ft_cot_t70_{}aug".format(aug):
                        completion_identifier = CompletionIdentifier("text-davinci-002", "zs_cot_t70", self.dataset_key)
                        completion_indices = list(range(aug))
                        break
                else:
                    raise NotImplementedError(self.preset_key)
            completion_dataset = CompletionDataset.load(completion_identifier)
            samples = completion_dataset.select_samples(sample_indices, completion_indices, only_correct=True)

        if samples is None:
            raise NotImplementedError(self.preset_key)

        # Format samples for model input/output.
        formatter = Formatter(self.model_type, self.prediction_template, dataset_key=self.dataset_key)
        formatted = formatter.format_samples(samples, include_labels=train)
        data = list_of_dicts_to_dict_of_lists(formatted)

        return data
