import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

import composer
from composer.utils import reproducibility
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


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune and prune a transformers model on a glue task.")
    
    # required arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        help="The configuration name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    # dataset, model, and tokenizer
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
             "dtype will be automatically derived from the model's weights."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=0,
        help="The percentage of the train set used as validation set in case there's no validation split provided.",
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
    parser.add_argument("--max_duration", type=str,   default="1ep", help="Total number of training epochs/batches/steps to perform.")

    # lr scheduler
    parser.add_argument(
        "--t_warmup", 
        type=str, 
        default="0.05dur", 
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--alpha_f",
        type=float, 
        default=0.0, 
        help="Final learning rate multiplier for the linear lr scheduler.")

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
    parser.add_argument("--deltaT",         type=int,   default=10,  help="The interval to mask weights.")

    # PMGP
    parser.add_argument("--sigma0",          type=float, default=1e-12, help="The smaller variance of the Mixture Gaussian prior.")
    parser.add_argument("--sigma1",          type=float, default=0.05,  help="The larger variance of the Mixture Gaussian orior.")
    parser.add_argument("--lambda_mix",      type=float, default=1e-7,  help="The mixing coefficient of the Mixture Gaussian prior.")
    parser.add_argument("--anneal_start",    type=int,   default=0,     help="The number of traing batches/steps for annealing to start.")
    parser.add_argument("--anneal_end",      type=int,   default=5000,  help="The number of traing batches/steps for annealing to end.")
    parser.add_argument("--masking_value",   type=float, default=0.0,   help="The masking value for the pruned weights.")
    parser.add_argument('--non_prior_name', 
                        nargs='+', 
                        type=str, 
                        default=["layernorm", "classifier", "pooler", "embedding"],
                        help="The names of the modules that should not be penalized by the prior.")

    # PLATON
    parser.add_argument("--beta1", type=float, default=0.85, help="The beta1 of PLATON pruner.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 of PLATON pruner.")

    # pruning algorithm
    parser.add_argument('--non_mask_name', 
                        nargs='+', 
                        type=str, 
                        default=["layernorm", "classifier", "pooler", "embedding"],
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

    # reproducibility
    reproducibility.seed_all(args.seed)

    # load the raw datasets
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
    )

    # load the model and tokenizer
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer,
    )
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    )



if __name__ == "__main__":
    main()