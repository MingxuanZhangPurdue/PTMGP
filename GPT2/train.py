import argparse
import warnings
import torch
from itertools import chain
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.distributed import barrier
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)


from composer.utils import reproducibility
from composer import Time, TimeUnit
from composer.utils.dist import get_sampler, get_local_rank
from composer.models.huggingface import HuggingFaceModel
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from composer import Trainer
from composer.callbacks import LRMonitor, RuntimeEstimator
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler
from pruners.PMGP import PMGP_Algorithm
from pruners.PLATON import PLATON_Algorithm


def parse_args():
    parser = argparse.ArgumentParser(description="Prune a GPT2 model with openwebtext dataset.")
    
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
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
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
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to download pretrained models and datasets.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        ),
    )
    parser.add_argument(
    "--validation_split_percentage",
    default=5,
    help="The percentage of the train set used as validation set in case there's no validation split",
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
        type=str, default=None, 
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
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--eval_interval", 
        type=str, 
        default="1ep", 
        help="Interval to evaluate the model."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=8, 
        help="Batch size (per device) for the evaluation dataloader."
    )

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
                        default=["layernorm", "bias"],
                        help="The names of the modules that should not be penalized by the prior.")

    # PLATON
    parser.add_argument("--beta1", type=float, default=0.85, help="The beta1 of PLATON pruner.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 of PLATON pruner.")

    # pruning algorithm
    parser.add_argument(
        '--non_mask_name',     
        nargs='+', 
        type=str, 
        default=["layernorm"],
        help="The names of the modules that should not be pruned."
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

    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",
            cache_dir=args.cache_dir,
            )
        raw_datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
            cache_dir=args.cache_dir,
        )

    # load the model and tokenizer
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
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
        trust_remote_code=args.trust_remote_code,
    )

    # resize the model if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # initialize the composer model
    metrics = [LanguageCrossEntropy(), LanguagePerplexity()]
    composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

    # get the task name
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    

    #if get_local_rank() > 0:
    #    print ("Waiting for main process to perform the mapping")
    #    barrier()

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    #if get_local_rank() == 0:
    #    print("Loading results from main process")
    #    barrier()

    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024


    if args.max_length is None:
        max_length = tokenizer.model_max_length
        if max_length > max_pos_embeddings:
            warnings.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using max_length={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --max_length xxx."
            )
            if max_pos_embeddings > 0:
                max_length = min(1024, max_pos_embeddings)
            else:
                max_length = 1024
    else:
        if args.max_length > tokenizer.model_max_length:
            warnings.warn(
                f"The max_length passed ({args.max_length}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using max_length={tokenizer.model_max_length}."
            )
        max_length = min(args.max_length, tokenizer.model_max_length)

    
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of max_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_length we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    #if get_local_rank() > 0:
    #    print ("Waiting for main process to perform the mapping")
    #    barrier()

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {max_length}",
    )

    #if get_local_rank() == 0:
    #    print("Loading results from main process")
    #    barrier()

    train_dataset = lm_datasets["train"]
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = lm_datasets["validation"]
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    train_sampler = get_sampler(train_dataset, shuffle=True)
    eval_sampler = get_sampler(eval_dataset, shuffle=False)


    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=default_data_collator, 
        batch_size=args.per_device_train_batch_size, 
        sampler=train_sampler
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=default_data_collator, 
        batch_size=args.per_device_eval_batch_size, 
        sampler=eval_sampler
    )

    # optimizer and learning rate scheduler creation
    optimizer = DecoupledAdamW(
        composer_model.parameters(), 
        lr=args.learning_rate, 
        betas=[0.9, 0.98], 
        eps=1.0e-06, 
        weight_decay=args.weight_decay
    )
    lr_scheduler = LinearWithWarmupScheduler(
        t_warmup=args.t_warmup, 
        alpha_f=args.alpha_f
    )

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
        eval_dataloader=eval_dataloader,
        eval_interval=args.eval_interval,

        # logging
        loggers=[wandb_logger],

        # callbacks
        callbacks=[LRMonitor(), RuntimeEstimator()],

        # algorithms
        #algorithms=[pruner_algorithm],

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