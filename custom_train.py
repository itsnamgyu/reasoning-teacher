"""
Run custom fine-tuning based experiments, i.e., fine-tuning models such as T5, and GPT-2 on GPUs.

Note, to check distributed errors used `TORCH_DISTRIBUTED_DEBUG=DETAIL`
Note, if deepspeed hangs at initialization, use `NCCL_P2P_DISABLE=1`. Thought, this seems to slow down the training a lot...
Note, to see more NCCL errors, use NCCL_DEBUG=WARN
"""
import argparse
import logging
import os

from custom.data_module import DataModule
from data.completion_dataset import CompletionMetadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from custom.model import Model

logging.basicConfig(level=logging.INFO)

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", type=str, default="multiarith")
    parser.add_argument("--model_key", type=str, default="t5_base")
    parser.add_argument("--train_key", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--preset_key", type=str, default="ft_cot")
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--disable_checkpointing", action="store_true")
    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    if args.precision == 16:
        args.precision = "bf16"
        print("Setting precision to bf16")

    dataset_key = args.dataset_key
    model_key = args.model_key
    train_key = args.train_key

    if "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # t5 tokenizers already append eos
    elif "t5" in model_key:
        hf_key = model_key.replace("_", "-")
        model = T5ForConditionalGeneration.from_pretrained(hf_key)
        tokenizer = T5TokenizerFast.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False
    elif "gpt2" in model_key:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        hf_key = model_key.replace("_", "-")
        tokenizer = GPT2Tokenizer.from_pretrained(hf_key)
        model = GPT2LMHeadModel.from_pretrained(hf_key)
        model_type = "decoder"
        append_eos = True
    else:
        raise NotImplementedError(model_key)

    if "ft_cot" in args.preset_key:
        completion_key = "ft_cot"
    elif args.preset_key == "ft":
        completion_key = "ft"
    elif args.preset_key == "fs_cot":
        raise NotImplementedError("We don't train models on fs_cot")
    else:
        raise NotImplementedError(args.preset_key)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = args.batch_size
    if args.inference_batch_size is None:
        inference_batch_size = batch_size
    else:
        inference_batch_size = args.inference_batch_size
    data_module = DataModule(dataset_key, args.preset_key, tokenizer, model_type, batch_size=batch_size,
                             inference_batch_size=inference_batch_size, num_workers=8, append_eos=append_eos)

    cm = CompletionMetadata(model_key, completion_key, dataset_key, data_module.finetune_key,
                            data_module.prediction_template, train_key=args.train_key)
    use_cpu_offload = args.strategy and "offload" in args.strategy
    lm = Model(model, tokenizer, model_type, use_cpu_offload=use_cpu_offload, completion_metadata=cm, lr=args.lr)

    if not os.path.exists("external_lightning_logs"):
        raise Exception("external_lightning_logs/ does not exist")
    default_root_dir = os.path.join("external_lightning_logs", "{}_{}_{}".format(model_key, dataset_key, train_key))
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                         default_root_dir=default_root_dir, min_epochs=20, max_epochs=20,
                         accumulate_grad_batches=args.accumulate, precision=args.precision,
                         enable_checkpointing=args.enable_checkpointing)

    trainer.fit(lm, datamodule=data_module)
