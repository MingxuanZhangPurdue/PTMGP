import argparse
import warnings

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
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

from pruners.MWA import MWA
from pruners.flexible_composer_lr_scheduler import LinearWithRewindsScheduler

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),
    "rte":  ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# add whether to use fixed mask during finla warmup
# combinatio between constant and linear with rewinds lr scheduler.

def str_int_and_none(value):
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If conversion to int fails, return the value as a string
        if value.casefold() == "none".casefold():
            return None
        else:
            return value
    
def float_and_none(value):
    try:
        # Try to convert the value to a float
        return float(value)
    except ValueError:
        # If conversion to float fails, return the value as a string
        if value.casefold() == "none".casefold():
            return None
        else:
            raise ValueError(f"Unsupported value type {value}")

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

    # dataset, model, and tokenizer
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
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
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            "sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load the checkpoint."
    )

    # evaluation
    parser.add_argument(
        "--eval_interval", 
        type=str_int_and_none, 
        default="1ep",
        help="Interval to evaluate the model."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=64,
        help="Batch size (per device) for the evaluation dataloader."
    )

    # training setups
    parser.add_argument(
        "--clipping_start",
        type=str_int_and_none,
        default=None,
        help="The start step to clip the gradients."
    )
    parser.add_argument(
        "--clipping_threshold",
        type=float_and_none,
        default=1.0,
        help="Gradient clipping threshold."
    )
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
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
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

    # lr scheduler type
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear_with_warmup",
        help="The lr scheduler to use.",
        choices=["linear_with_warmup", "linear_with_rewinds"],
    )
    # linear with warmup lr scheduler specificaions
    parser.add_argument(
        "--t_warmup", 
        type=str, 
        default="0.01dur",
        help="Time for the warmup in the linear with warmup lr scheduler."
    )
    parser.add_argument(
        "--alpha_i",
        type=float,
        default=1.0, 
        help="Initial learning rate multiplier in the linear with warmup lr scheduler."
    )
    parser.add_argument(
        "--alpha_f",
        type=float,
        default=0.0, 
        help="Final learning rate multiplier in the linear with warmup lr scheduler."
    )
    # linear with rewinds lr scheduler specificaions
    parser.add_argument(
        "--t_iw",
        type=str,
        default="2ep",
        help="Time for the initial warmup in the linear with rewinds lr scheduler."
    )
    parser.add_argument(
        "--t_rewind",
        type=str,
        default="6ep",
        help="Time for each rewind in the linear with rewinds lr scheduler."
    )
    parser.add_argument(
        "--t_fw",
        type=str,
        default="2ep",
        help="Time for the final warmup in the linear with rewinds lr scheduler."
    )
    parser.add_argument(
        "--alpha_i_iw",
        type=float,
        default=1.0, 
        help="Initial learning rate multiplier in the linear with rewinds lr scheduler during inital warmup stage."
    )
    parser.add_argument(
        "--alpha_f_iw",
        type=float,
        default=0.0, 
        help="Final learning rate multiplier in the linear with rewinds lr scheduler during inital warmup stage."
    )
    parser.add_argument(
        "--alpha_i_rewind",
        type=float,
        default=1.0, 
        help="Initial learning rate multiplier in the linear with rewinds lr scheduler during rewind stage."
    )
    parser.add_argument(
        "--alpha_f_rewind",
        type=float,
        default=0.0, 
        help="Final learning rate multiplier in the linear with rewinds lr scheduler during rewind stage."
    )
    parser.add_argument(
        "--alpha_i_fw",
        type=float,
        default=1.0, 
        help="Initial learning rate multiplier in the linear with rewinds lr scheduler during final warmup stage."
    )
    parser.add_argument(
        "--alpha_f_fw",
        type=float,
        default=0.0, 
        help="Final learning rate multiplier in the linear with rewinds lr scheduler during final warmup stage."
    )
    parser.add_argument(
        "--num_rewinds",
        type=int,
        default=1,
        help="Number of rewinds in the linear with rewinds lr scheduler."
    )

    # wandb logging
    parser.add_argument(
        "--wandb_project_name", 
        type=str, 
        default=None, 
        help="The name of the wandb project to log to."
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str, 
        default=None, 
        help="The wandb run name."
    )
    
    # reproducibility
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed to use for reproducibility."
    )

    # pruning scheduler
    parser.add_argument(
        "--initial_sparsity",      
        type=float,            
        default=0.0,     
        help="The initial sparsity of the model."
    )
    parser.add_argument(
        "--final_sparsity",        
        type=float,            
        default=0.0,   
        help="The final sparsity of the model."
    )
    parser.add_argument(
        "--initial_warmup_steps",    
          type=str_int_and_none,   
          default=0, 
          help="The number of training batches/steps for initial warmup."
    )
    parser.add_argument(
        "--sparse_finetune_steps",       
        type=str_int_and_none,   
        default=0,     
        help="The number of training batches/steps for sparse finetuning."
    )
    parser.add_argument(
        "--pruning_interval",             
        type=str_int_and_none,   
        default=10,    
        help="The number of training steps between two pruning operations."
    )

    # MWA
    parser.add_argument(
        "--sigma0",             
        type=float,            
        default=1e-15, 
        help="The base value of the sigma0."
    )
    parser.add_argument(
        "--sigma1",             
        type=float,            
        default=0.1,   
        help="The base value of the sigma1."
    )
    
    parser.add_argument(
        "--lambda_mix",         
        type=float,            
        default=1e-1,  
        help="The base value of the lambda_mix."
    )
    parser.add_argument(
        "--alpha_i_lambda_mix",
        type=float,
        default=1.0,
        help="The initial factor value of the lambda_mix."
    )
    parser.add_argument(
        "--alpha_f_lambda_mix",     
        type=float,            
        default=1.0,   
        help="The final factor value of the lambda_mix."
    )
    parser.add_argument(
        "--anneal_start_lambda_mix",
        type=str_int_and_none,
        default=None,
        help="The start step to anneal the lambda_mix."
    )
    parser.add_argument(
        "--anneal_end_lambda_mix",
        type=str_int_and_none,
        default=None,
        help="The end step to anneal the lambda_mix."
    )

    # logging interval for GBReg
    parser.add_argument(
        "--log_interval",
        type=str_int_and_none,
        default=None,
        help="Interval to log all research-related information."
    )

    # pruning configurations
    parser.add_argument(
        '--pruning_params', 
        nargs='+', 
        type=str, 
        default=[ 
            "bert.encoder.layer.*.attention.self.query.weight",
            "bert.encoder.layer.*.attention.self.key.weight",
            "bert.encoder.layer.*.attention.self.value.weight",
            "bert.encoder.layer.*.attention.output.dense.weight",
            "bert.encoder.layer.*.intermediate.dense.weight",
            "bert.encoder.layer.*.output.dense.weight",
        ],
        help="The names of the modules that should be pruned. We will match the names using regex."
    )

    args = parser.parse_args()

    return args


def main():

    # parse the arguments
    args = parse_args()

    # reproducibility
    reproducibility.seed_all(args.seed)

    # load the raw datasets
    raw_datasets = load_dataset(
        "glue",
        args.task_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
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
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )

    # set the evluation metrics based on the task
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

    if args.max_seq_length > tokenizer.model_max_length:
        warnings.warn(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)

        if "label" in examples:
            # in all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # dataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        if args.precision == "amp_fp16" or args.precision == "amp_bf16":
            use_fp16 = True
        else:
            use_fp16 = False
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if use_fp16 else None))

    train_sampler = get_sampler(train_dataset, shuffle=True)
    eval_sampler = get_sampler(eval_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, sampler=train_sampler)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, sampler=eval_sampler)

    if args.task_name == "mnli":
        mm_eval_dataset = processed_datasets["validation_mismatched"]
        mm_eval_sampler = get_sampler(mm_eval_dataset, shuffle=False)
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

    # optimizer and learning rate scheduler creation
    optimizer = DecoupledAdamW(
        composer_model.parameters(), 
        lr=args.learning_rate, 
        betas=[0.9, 0.98], 
        eps=1.0e-06, 
        weight_decay=args.weight_decay
    )
    if args.lr_scheduler == "linear_with_warmup":
        lr_scheduler = LinearWithWarmupScheduler(
            t_warmup=args.t_warmup,
            alpha_i=args.alpha_i,
            alpha_f=args.alpha_f
        )
    elif args.lr_scheduler == "linear_with_rewinds":
        lr_scheduler = LinearWithRewindsScheduler(
            t_iw=args.t_iw,
            t_rewind=args.t_rewind,
            t_fw=args.t_fw,
            alpha_i_iw=args.alpha_i_iw,
            alpha_f_iw=args.alpha_f_iw,
            alpha_i_rewind=args.alpha_i_rewind,
            alpha_f_rewind=args.alpha_f_rewind,
            alpha_i_fw=args.alpha_i_fw,
            alpha_f_fw=args.alpha_f_fw,
            num_rewinds=args.num_rewinds
        )
    else:
        raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")

    # initialize the wandb logger
    wandb_logger = WandBLogger(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
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
    
    pruner_algorithm = MWA.from_args(train_size, max_train_steps, len(train_dataloader), args)

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
        load_path=args.load_path,

        # reproducibility
        seed=args.seed,
    )

    # Train
    trainer.fit()

if __name__ == "__main__":
    main()

