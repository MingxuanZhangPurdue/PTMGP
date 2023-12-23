import argparse
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

import composer
from composer.utils import dist
from composer.models.huggingface import HuggingFaceModel
from torchmetrics.classification import MulticlassAccuracy, MulticlassMatthewsCorrCoef, MulticlassF1Score
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef
from composer import Trainer
from composer.loggers import WandBLogger, FileLogger, ProgressBarLogger


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
    parser = argparse.ArgumentParser(description="Finetune and prune a transformers model on a glue task.")
    
    # required arguments
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

    # dataset, model, and tokenizer
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

    # checkpointing
    parser.add_argument("--run_name", type=str, default=None, help="Name of the run.")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save the checkpoints.")
    parser.add_argument("--save_filename", type=str, default='ep{epoch}-ba{batch}-rank{rank}.pt', help="Filename to save the checkpoints.")
    parser.add_argument("--save_interval", type=str, default="1ep", help="Interval to save the checkpoints.")
    parser.add_argument("--save_latest_filename", type=str, default='latest-rank{rank}.pt', help="Filename to save the last checkpoint.")
    parser.add_argument("--autoresume", action="store_true", help="If passed, will resume the latest checkpoint if any.")
    parser.add_argument("--save_overwrite", action="store_true", help="If passed, will overwrite the checkpoints if any.")


    # evaluation
    parser.add_argument("--eval_interval", type=str, default="1ep", help="Interval to evaluate the model.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")

    # training setups
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
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--max_duration", type=str, default="1ep", help="Total number of training epochs/batches/steps to perform.")
    parser.add_argument("--t_warmup", type=str, default="1ba", help="Number of steps for the warmup in the lr scheduler.")

    # logging
    parser.add_argument("--loggers", type=str, nargs='+', default=[], help="Loggers to use.")
    parser.add_argument("--log_to_console", action="store_true", help="If passed, will log to console.")

    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use for reproducibility.")

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

    # get the metrics
    if args.task_name == "stsb":
        metrics = [PearsonCorrCoef(), SpearmanCorrCoef()]
    elif args.task_name == "cola":
        metrics = [MulticlassMatthewsCorrCoef(num_classes=num_labels)]
    elif args.task_name == "mrpc" or args.task_name == "qqp":
        metrics = [MulticlassAccuracy(num_classes=num_labels, average='micro'), MulticlassF1Score(num_classes=num_labels, average='micro')]
    else:
        metrics = [MulticlassAccuracy(num_classes=num_labels, average='micro')]

    # wrap the model
    composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

    # preprocess the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # padding strategy
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            # in all cases, rename the column to labels because the model will expect that.
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

    # dataLoaders creation:
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

    # optimizer and lr_scheduler creation
    optimizer = torch.optim.AdamW(composer_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = composer.optim.LinearWithWarmupScheduler(t_warmup=args.t_warmup, t_max=args.max_duration)


    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        max_duration=args.max_duration,
        device_train_microbatch_size='auto',
        device='gpu' if torch.cuda.is_available() else 'cpu',
        precision=args.precision,
        schedulers=lr_scheduler,
        eval_interval=args.eval_interval,

        # logging
        loggers=None,

        # callbacks
        callbacks=None,

        # algorithms
        algorithms=None,

        # checkpointing
        run_name=args.run_name,
        save_folder=args.save_folder,
        save_filename=args.save_filename,
        save_interval=args.save_interval,
        save_latest_filename=args.save_latest_filename,
        save_overwrite=args.save_overwrite,
        autoresume=args.autoresume,

        # reproducibility
        seed=args.seed,
    )

    # Train
    trainer.fit()



if __name__ == "__main__":
    main()

