import argparse
from dataclasses import dataclass, field
from typing import Optional

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
from composer.core import Evaluator
from composer import Time, TimeUnit
from composer.utils import dist
from composer.models.huggingface import HuggingFaceModel
from torchmetrics.classification import MulticlassAccuracy, MulticlassMatthewsCorrCoef, MulticlassF1Score
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef
from composer import Trainer
from composer.callbacks import LRMonitor, RuntimeEstimator
from composer.loggers import WandBLogger
from pruners.PMGP import PMGP_Algorithm
from pruners.PLATON import PLATON_Algorithm



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
    parser.add_argument("--run_name", 
                        type=str, 
                        default=None, 
                        help="Name of the run.")
    parser.add_argument("--save_folder", 
                        type=str, default=None, 
                        help="Folder to save the checkpoints.")
    parser.add_argument("--save_interval", 
                        type=str, 
                        default="1ep", 
                        help="Interval to save the checkpoints.")
    parser.add_argument("--autoresume", 
                        action="store_true", 
                        help="If passed, will resume the latest checkpoint if any.")
    parser.add_argument("--save_overwrite", 
                        action="store_true", 
                        help="If passed, will overwrite the checkpoints if any.")
    parser.add_argument("--save_latest_filename", 
                        type=str, 
                        default='latest-rank{rank}.pt', 
                        help="Filename to save the last checkpoint.")
    parser.add_argument("--save_filename", 
                        type=str, 
                        default='ep{epoch}-ba{batch}-rank{rank}.pt', help="Filename to save the checkpoints.")

    # evaluation
    parser.add_argument("--eval_interval", 
                        type=str, 
                        default="1ep", 
                        help="Interval to evaluate the model.")
    parser.add_argument("--per_device_eval_batch_size", 
                        type=int, 
                        default=8, 
                        help="Batch size (per device) for the evaluation dataloader.")

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
    parser.add_argument("--weight_decay", type=float, default=0.0,  help="Weight decay to use.")
    parser.add_argument("--max_duration", type=str,  default="1ep", help="Total number of training epochs/batches/steps to perform.")
    parser.add_argument("--t_warmup",     type=str,  default="1ba", help="Number of steps for the warmup in the lr scheduler.")

    # wandb logging
    parser.add_argument("--wandb_project", type=str, default=None, help="The wandb project to log to.")
    parser.add_argument("--wandb_name",    type=str, default=None, help="The wandb run name.")
    
    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use for reproducibility.")

    # cubic pruning scheduler
    parser.add_argument("--final_ratio",    type=float, default=0.1, help="The final ratio of the remaining weights.")
    parser.add_argument("--initial_ratio",  type=float, default=1,   help="The initial ratio of the remaining weights.")
    parser.add_argument("--initial_warmup", type=int,   default=1,   help="The number of training batches/steps for initial warmup.")
    parser.add_argument("--final_warmup",   type=int,   default=1,   help="The number of training batches/steps for final warmup.")
    parser.add_argument("--warmup_steps",   type=int,   default=100, help="The number of warmup steps.")
    parser.add_argument("--deltaT",         type=int,   default=10,  help="The interval to mask weights.")

    # PMGP
    parser.add_argument("--sigma0",          type=float, default=1e-12, help="The smaller variance of the Mixture Gaussian prior.")
    parser.add_argument("--sigma1",          type=float, default=0.05,  help="The larger variance of the Mixture Gaussian orior.")
    parser.add_argument("--lambda_mix",      type=float, default=1e-7,  help="The mixing coefficient of the Mixture Gaussian prior.")
    parser.add_argument("--anneal_start",    type=int,   default=0,     help="The number of traing batches/steps for annealing to start.")
    parser.add_argument("--anneal_end",      type=int,   default=5000,  help="The number of traing batches/steps for annealing to end.")
    parser.add_argument("--masking_value",   type=float, default=0.0,   help="The masking value for the pruned weights.")
    parser.add_argument(
        "--apply_prior_on_all_layers",
        action="store_true",
        help="If passed, will apply the prior on all layers.",
    )

    # PLATON
    parser.add_argument("--beta1", type=float, default=0.85, help="The beta1 of PLATON pruner.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 of PLATON pruner.")

    # pruning algorithm
    parser.add_argument('--non_mask_name', 
                        nargs='+', 
                        type=str, 
                        default=["layernorm", "classifier", "pooler"], 
                        help="The names of the modules that should not be pruned.")
    parser.add_argument("--pruner", 
                        type=str, 
                        default="PMGP", 
                        help="The pruner to use.", 
                        choices=["PMGP", "PLATON"])

    args = parser.parse_args()
    return args


def main():

    # parse the arguments
    args = parse_args()

    # load the raw datasets
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

    # preprocess the raw datasets
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

    if args.task_name == "mnli":
        mm_eval_dataset = processed_datasets["validation_mismatched"]
        mm_eval_sampler = dist.get_sampler(mm_eval_dataset, shuffle=False)
        mm_eval_dataloader = DataLoader(mm_eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, sampler=mm_eval_sampler)
        mnli_matched_task = Evaluator(
            label='mnli_matched_accuracy',
            dataloader=eval_dataloader,
            metric_names=['MulticlassAccuracy']
        )
        mnli_mismatched_task = Evaluator(
            label='mnli_mismatched_accuracy',
            dataloader=mm_eval_dataloader,
            metric_names=['MulticlassAccuracy']
        )

    # optimizer and lr_scheduler creation
    optimizer = torch.optim.AdamW(composer_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = composer.optim.LinearWithWarmupScheduler(t_warmup=args.t_warmup, t_max=args.max_duration)

    # initialize the wandb logger
    wandb_logger = WandBLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        init_kwargs = {"config": vars(args)}
    )

    # initialize the pruner algorithm
    train_size = len(train_dataset)
    train_time = Time.from_timestring(args.max_duration)
    if train_time.unit == TimeUnit.EPOCH:
        max_train_steps = len(train_dataloader) * train_time.value
    elif train_time.unit == TimeUnit.BATCH:
        max_train_steps = train_time.value
    else:
        raise ValueError(f"Unsupported time unit: {train_time.unit}")
    
    if args.pruner == "PMGP":
        pruner_algorithm = PMGP_Algorithm.from_args(train_size, max_train_steps, args)
    elif args.pruner == "PLATON":
        pruner_algorithm = PLATON_Algorithm.from_args(max_train_steps, args)

    # initialize the trainer
    trainer = Trainer(
        # training
        model=composer_model,
        train_dataloader=train_dataloader,
        optimizers=optimizer,
        max_duration=args.max_duration,
        device_train_microbatch_size='auto',
        device='gpu' if torch.cuda.is_available() else 'cpu',
        precision=args.precision,
        schedulers=lr_scheduler,

        # evaluation
        eval_dataloader=[mnli_matched_task, mnli_mismatched_task] if args.task_name == "mnli" else eval_dataloader,
        eval_interval=args.eval_interval,

        # logging
        loggers=[wandb_logger],

        # callbacks
        callbacks=[LRMonitor(), RuntimeEstimator()],

        # algorithms
        algorithms=[pruner_algorithm],

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

