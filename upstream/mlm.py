import argparse
import torch
from torch.utils.data import DataLoader

from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from composer.utils.dist import get_sampler
from composer.utils import reproducibility
from composer import Time, TimeUnit
from composer.models.huggingface import HuggingFaceModel
from composer.algorithms import GradientClipping
from composer import Trainer
from composer.callbacks import LRMonitor, RuntimeEstimator
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy

from pruners.BReg import BReg
from pruners.utils_composer import LinearWithRewindsScheduler
from upstream.utils_datasets import get_tokenized_mlm_datasets

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def my_custom_type(value):
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If conversion to int fails, return the value as a string
        return value

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
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
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Where to download pretrained models and datasets.",
    )
    parser.add_argument(
        "--overwrite_cache", 
        action="store_true", 
        help="Overwrite the cached training and evaluation sets"
    )

    # model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="The specific model version to use "
        "(can be a branch name, tag name or commit id).",
    )

    # datasets arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_name_2",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name_2",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )

    # datasets preprocessing arguments
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--mlm_probability", 
        type=float, 
        default=0.15, 
        help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    # training arguments
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="amp_fp16",
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
        default=5e-4,
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
        default="3ep",
        help="Total number of training epochs/batches/steps to perform."
    )
    parser.add_argument(
        "--t_warmup", 
        type=str, 
        default="0.05dur", 
        help="Number of steps for the warmup in the linear lr scheduler."
    )
    parser.add_argument(
        "--alpha_f",
        type=float, 
        default=0.01, 
        help="Final learning rate multiplier for the linear lr scheduler."
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
        default=128,
        help="Batch size (per device) for the evaluation dataloader."
    )

    # wandb logging
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="upstream_bert_base", 
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
        default=42, 
        help="A seed for reproducible training."
    )

    # cubic pruning scheduler
    parser.add_argument("--final_ratio",        type=float,            default=0.1, help="The final ratio of the remaining weights.")
    parser.add_argument("--initial_ratio",      type=float,            default=1,   help="The initial ratio of the remaining weights.")
    parser.add_argument("--initial_warmup",     type=my_custom_type,   default=1,   help="The number of training batches/steps for initial warmup.")
    parser.add_argument("--final_warmup",       type=my_custom_type,   default=0,   help="The number of training batches/steps for final warmup.")
    parser.add_argument("--deltaT",             type=my_custom_type,   default=10,  help="The interval to mask weights.")
    parser.add_argument("--deltaT_cooldown",    type=my_custom_type,   default=10,  help="The interval to mask weights.")
    parser.add_argument("--sparse_fine_tune",   type=my_custom_type,   default=0,   help="The number of training batches/steps for sparse fine-tuning.")

    # BReg
    parser.add_argument("--sigma0",             type=float,            default=1e-13, help="The smaller variance of the Mixture Gaussian prior.")
    parser.add_argument("--alpha_i_sigma0",     type=float,            default=1.0,   help="The initial factor value of the sigma0.")
    parser.add_argument("--alpha_f_sigma0",     type=float,            default=1.0,   help="The final factor value of the sigma0.")

    parser.add_argument("--sigma1",             type=float,            default=0.05,   help="The larger variance of the Mixture Gaussian prior.")
    parser.add_argument("--alpha_i_sigma1",     type=float,            default=1.0,   help="The initial factor value of the sigma1.")
    parser.add_argument("--alpha_f_sigma1",     type=float,            default=1.0,   help="The final factor value of the sigma1.")
    
    parser.add_argument("--lambda_mix",         type=float,            default=1e-3,  help="The mixing coefficient of the Mixture Gaussian prior.")
    parser.add_argument("--alpha_i_lambda",     type=float,            default=1.0,   help="The initial factor value of the lambda_mix.")
    parser.add_argument("--alpha_f_lambda",     type=float,            default=0.01,  help="The final factor value of the lambda_mix.")

    parser.add_argument("--anneal_start",       type=my_custom_type,   default=None,  help="The number of traing batches/steps for lambda_mix annealing to start.")
    parser.add_argument("--anneal_end",         type=my_custom_type,   default=None,  help="The number of traing batches/steps for lambda_mix annealing to end.")

    parser.add_argument("--masking_value",      type=float,            default=0.0,   help="The filling value of the masked weights.")

    parser.add_argument('--non_prior_name',
                        type=str,
                        default=None,
                        nargs='+',
                        help="The names of the modules that should not be penalized by the prior, if any. We will match the names using regex.")

    # pruning algorithm selection
    parser.add_argument(
        '--non_mask_name', 
        nargs='+', 
        type=str, 
        default=["layernorm", "classifier", "pooler", "embedding", "bias", "prediction"],
        help="The names of the modules that should not be pruned. We will match the names using regex."
    )
    args = parser.parse_args()

    return args


def main():

    # parse the arguments
    args = parse_args()

    # reproducibility
    reproducibility.seed_all(args.seed)

    # load the model and tokenizer
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = get_tokenized_mlm_datasets(
        tokenizer=tokenizer,
        args=args,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    if args.precision == "amp_fp16" or args.precision == "amp_bf16":
        use_fp16 = True
    else:
        use_fp16 = False

    pad_to_multiple_of_8 = (
        args.line_by_line
        and use_fp16
        and not args.pad_to_max_length
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    train_sampler = get_sampler(train_dataset, shuffle=True)
    eval_sampler = get_sampler(eval_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size, 
        sampler=train_sampler
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size, 
        sampler=eval_sampler
    )

    # wrap the model with the composer model
    metrics = [
        LanguageCrossEntropy(ignore_index=-100),
        MaskedAccuracy(ignore_index=-100)
    ]
    composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

    # optimizer and learning rate scheduler creation
    optimizer = DecoupledAdamW(
        composer_model.parameters(), 
        lr=args.learning_rate, 
        betas=[0.9, 0.999], 
        eps=1e-8,
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
    pruner_algorithm = PMGP_Algorithm.from_args(train_size, max_train_steps, args)

    # gradient clipping following oBERT
    gc = GradientClipping(clipping_type='norm', clipping_threshold=args.max_grad_norm)

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
        algorithms=[gc, pruner_algorithm],

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