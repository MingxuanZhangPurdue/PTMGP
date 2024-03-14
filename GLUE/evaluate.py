import argparse
import warnings
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, DataCollatorWithPadding

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "qqp":  ("question1", "question2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pruned model a glue task.")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
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
    
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    if args.task_name == "mnli":
        split = "validation_matched"
    elif args.task_name == "mnli_mismatched":
        split = "validation_mismatched"
    else:
        split = "validation"

    raw_dataset = load_dataset(
        "glue",
        args.task_name,
        split=split,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    label_list = raw_dataset.features["label"].names
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path if args.tokenizer_name_or_path is None else args.tokenizer_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

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
        result = tokenizer(*texts, padding=False, max_length=max_seq_length, truncation=True)

        if "label" in examples:
            # in all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    eval_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    y_hat = []
    y = []
    model = model.to("cuda") if torch.cuda.is_available() else model
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        y_hat.append(outputs.logits.argmax(dim=-1))
        y.append(batch["labels"])
    y_hat = torch.cat(y_hat)
    y = torch.cat(y)

    if args.task_name in ["mnli", "mnli_mismatched"]:
        metric = load("mnli")
    else:
        metric = load(args.task_name)
    result = metric.compute(predictions=y_hat, references=y)
    print(result)

if __name__ == "__main__":
    main()
