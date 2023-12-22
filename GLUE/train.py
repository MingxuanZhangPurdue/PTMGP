import argparse
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

from composer.utils import dist
from composer.models.huggingface import HuggingFaceModel
from torchmetrics.classification import MulticlassAccuracy, MulticlassMatthewsCorrCoef, MulticlassF1Score
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune and prune a transformers model on a glue task")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to download pretrained models and datasets.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="The precision to use, can be fp32, amp_fp16, or amp_bf16.",
        choices=[None, "fp32", "amp_fp16", "amp_bf16"],
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="LinearWithWarmupScheduler",
        help="The scheduler type to use.",
        choices=["MultiStepWithWarmupScheduler", "LinearWithWarmupScheduler", "CosineAnnealingWithWarmupScheduler"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    args = parser.parse_args()
    return args




def main():

    # parse the arguments
    args = parse_args()

    # load the dataset
    raw_datasets = load_dataset(
        "glue",
        args.task_name,
        cache_dir=args.cache_dir,
    )
    
    # determine the number of labels
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # load the model and tokenizer
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    # Get the metrics
    if args.task_name == "stsb":
        metrics = [PearsonCorrCoef(), SpearmanCorrCoef()]
    elif args.task_name == "cola":
        metrics = [MulticlassMatthewsCorrCoef(num_classes=num_labels)]
    elif args.task_name == "mrpc" or args.task_name == "qqp":
        metrics = [MulticlassAccuracy(num_classes=num_labels, average='micro'), MulticlassF1Score(num_classes=num_labels, average='micro')]
    else:
        metrics = [MulticlassAccuracy(num_classes=num_labels, average='micro')]

    # Wrap the model
    composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics)

    #preprocess the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # Padding strategy
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result
    

    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        if args.precision == "amp_fp16" or args.precision == "amp_bf16":
            use_fp16 = True
        else:
            use_fp16 = False
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if use_fp16 else None))

    train_sampler = dist.get_sampler(train_dataset, shuffle=True)
    eval_sampler = dist.get_sampler(eval_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, sampler=train_sampler)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, sampler=eval_sampler)

    # Optimizer
    optimizer = torch.optim.AdamW(composer_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

