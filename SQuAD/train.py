import argparse
import warnings

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

from torchmetrics.classification import MulticlassAccuracy, MulticlassMatthewsCorrCoef, MulticlassF1Score
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef

from composer.utils.dist import get_sampler
from composer.utils import reproducibility
from composer.core import Evaluator
from composer import Time, TimeUnit
from composer.models.huggingface import HuggingFaceModel
from composer import Trainer
from composer.callbacks import LRMonitor, RuntimeEstimator
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler

from utils_qa import postprocess_qa_predictions
from pruners.PMGP import PMGP_Algorithm
from pruners.PLATON import PLATON_Algorithm

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune and prune a transformers model on a glue task.")

    # dataset, model, and tokenizer
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="squad",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
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
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Where to download pretrained models and datasets.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            "sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )

    # checkpointing
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None, 
        help="Name of the run."
    )
    parser.add_argument(
        "--save_folder", 
        type=str, 
        default=None, 
        help="Folder to save the checkpoints."
    )
    parser.add_argument(
        "--save_interval",     
        type=str, 
        default="1ep", 
        help="Interval to save the checkpoints."
    )
    parser.add_argument(
        "--autoresume", 
        action="store_true", 
        help="If passed, will resume the latest checkpoint if any."
    )
    parser.add_argument(
        "--save_overwrite", 
        action="store_true", 
        help="If passed, will overwrite the checkpoints if any."
    )
    parser.add_argument(
        "--save_latest_filename", 
        type=str, 
        default='latest-rank{rank}.pt', 
        help="Filename to save the last checkpoint."
    )
    parser.add_argument(
        "--save_filename", 
        type=str, 
        default='ep{epoch}-ba{batch}-rank{rank}.pt', help="Filename to save the checkpoints."
    )

    # evaluation
    parser.add_argument(
        "--eval_interval", 
        type=str, 
        default="0.1dur", 
        help="Interval to evaluate the model."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=256,
        help="Batch size (per device) for the evaluation dataloader."
    )

    # training setups
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="The precision to use, can be fp32, amp_fp16, or amp_bf16.",
        choices=[None, "fp32", "amp_fp16", "amp_bf16"],
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0,  
        help="Weight decay to use."
    )
    parser.add_argument(
        "--max_duration", 
        type=str,   
        default="10ep",
        help="Total number of training epochs/batches/steps to perform."
    )

    # lr scheduler
    parser.add_argument(
        "--t_warmup", 
        type=str, 
        default="0.05dur", 
        help="Number of steps for the warmup for the linear lr scheduler."
    )
    parser.add_argument(
        "--alpha_f",
        type=float, 
        default=0.0, 
        help="Final learning rate multiplier for the linear lr scheduler."
    )

    # wandb logging
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default=None, 
        help="The wandb project to log to."
    )
    parser.add_argument(
        "--wandb_name", 
        type=str, 
        default=None, 
        help="The wandb run name."
    )
    
    # reproducibility
    parser.add_argument(
        "--seed", 
        type=int, 
        default=47, 
        help="Random seed to use for reproducibility."
    )

    # cubic pruning scheduler
    parser.add_argument("--final_ratio",        type=float, default=0.2, help="The final ratio of the remaining weights.")
    parser.add_argument("--initial_ratio",      type=float, default=1,   help="The initial ratio of the remaining weights.")
    parser.add_argument("--initial_warmup",     type=int,   default=5400,   help="The number of training batches/steps for initial warmup.")
    parser.add_argument("--final_warmup",       type=int,   default=22000,   help="The number of training batches/steps for final warmup.")
    parser.add_argument("--deltaT",             type=int,   default=10,  help="The interval to mask weights.")

    # PMGP
    parser.add_argument("--sigma0",             type=float, default=1e-15, help="The smaller variance of the Mixture Gaussian prior.")
    parser.add_argument("--sigma1",             type=float, default=0.1,   help="The larger variance of the Mixture Gaussian prior.")
    parser.add_argument("--lambda_mix",         type=float, default=1e-3,  help="The mixing coefficient of the Mixture Gaussian prior.")
    parser.add_argument("--alpha_i_lambda",     type=float, default=1.0,   help="The initial factor value of the lambda_mix.")
    parser.add_argument("--alpha_f_lambda",     type=float, default=0.01,  help="The final factor value of the lambda_mix.")
    parser.add_argument("--anneal_start_lambda",type=int,   default=None,  help="The number of traing batches/steps for lambda_mix annealing to start.")
    parser.add_argument("--anneal_end_lambda",  type=int,   default=None,  help="The number of traing batches/steps for lambda_mix annealing to end.")
    parser.add_argument("--masking_value",      type=float, default=0.0,   help="The masking value for the pruned weights.")
    parser.add_argument('--non_prior_name',     
                        type=str,
                        default=["layernorm", "classifier", "pooler", "embedding", "bias"],
                        nargs='+',
                        help="The names of the modules that should not be penalized by the prior.")

    # PLATON
    parser.add_argument("--beta1", type=float, default=0.85, help="The beta1 of PLATON pruner.")
    parser.add_argument("--beta2", type=float, default=0.975, help="The beta2 of PLATON pruner.")

    # pruning algorithm selection
    parser.add_argument(
        '--non_mask_name', 
        nargs='+', 
        type=str, 
        default=["layernorm", "classifier", "pooler", "embedding"],
        help="The names of the modules that should not be pruned. We will match the names using regex."
    )
    parser.add_argument(
        "--pruner", 
        type=str, 
        default="PMGP", 
        help="The pruner to use.", 
        choices=["PMGP", "PLATON"]
    )

    args = parser.parse_args()

    return args
