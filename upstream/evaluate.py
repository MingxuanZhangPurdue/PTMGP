import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from evaluate import load
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from upstream.utils_datasets import get_tokenized_mlm_datasets

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a upstream pruned model.")
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
        "--checkpoint_load_path",
        type=str,
        default=None,
        help="Path to load the checkpoint."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=128,
        help="Batch size (per device) for the evaluation dataloader."
    )
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

    args = parser.parse_args()

    return args


def main():

    # parse the arguments
    args = parse_args()

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

    model_state_dict = torch.load(args.checkpoint_load_path)["state"]["model"] if args.checkpoint_load_path is not None else None
    if model_state_dict is not None:
        for key in list(model_state_dict.keys()):
            parts = key.split('.')
            new_key = '.'.join(parts[1:])
            model_state_dict[new_key] =  model_state_dict.pop(key)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        state_dict = model_state_dict
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

    eval_dataset = tokenized_datasets["validation"]
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=None,
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size, 
        shuffle=False
    )


    model = model.to("cuda") if torch.cuda.is_available() else model
    model.eval()
    metric = load("accuracy")
    targets = []
    predictions = []
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        targets.append(batch["labels"].cpu().view(-1))
        predictions.append(outputs.logits.argmax(dim=-1).cpu().view(-1))
    targets = torch.cat(targets)
    predictions = torch.cat(predictions)
    indices = torch.where(targets != -100)
    result = metric.compute(predictions=predictions[indices], references=targets[indices])
    print (result)

if __name__ == "__main__":
    main()