import warnings
from itertools import chain
from datasets import concatenate_datasets, load_dataset
from composer.utils.dist import barrier, get_local_rank

def _get_tokenized_mlm_datasets_from_raw_datasets(
    raw_datasets,
    tokenizer,
    args,
):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            warnings.warn(
                "The tokenizer picked seems to have a very large `model_max_length`"
                f"({tokenizer.model_max_length}). Picking 1024 instead. You can "
                "change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            warnings.warn(
                f"The max_seq_length passed ({args.max_seq_length})"
                "is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using "
                f"max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line
                for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see
                # below) is more efficient when it receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        if get_local_rank() > 0:
            print("Waiting for main process to perform the mapping")
            barrier()
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset line_by_line",
        )
        if get_local_rank() == 0:
            print("Loading results from main process")
            barrier()
    else:
        # Otherwise, we tokenize every text, then concatenate them together before
        # splitting them in smaller parts. We use `return_special_tokens_mask=True`
        # because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        if get_local_rank() > 0:
            print("Waiting for main process to perform the mapping")
            barrier()
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        if get_local_rank() == 0:
            print("Loading results from main process")
            barrier()

        # Main data processing function that will concatenate all texts from our
        # dataset and generate chunks of max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported
            # it instead of this drop, you can customize this part to your needs
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result
        
        if get_local_rank() > 0:
            print("Waiting for main process to perform the mapping")
            barrier()
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )
        if get_local_rank() == 0:
            print("Loading results from main process")
            barrier()
    return tokenized_datasets


def _get_mlm_raw_datasets(args):
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        raw_datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    if args.dataset_name_2 is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets_2 = load_dataset(
            args.dataset_name_2,
            args.dataset_config_name_2,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        if "validation" not in raw_datasets_2.keys():
            raw_datasets_2["validation"] = load_dataset(
                args.dataset_name_2,
                args.dataset_config_name_2,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
            )
            raw_datasets_2["train"] = load_dataset(
                args.dataset_name_2,
                args.dataset_config_name_2,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
            )

        for split in ["validation", "train"]:
            raw_datasets[split] = concatenate_datasets(
                [raw_datasets[split], raw_datasets_2[split]]
            )
    return raw_datasets

def get_tokenized_mlm_datasets(
    args,
    tokenizer,
):
    raw_datasets = _get_mlm_raw_datasets(args=args)
    tokenized_datasets = _get_tokenized_mlm_datasets_from_raw_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        args=args,
    )
    return tokenized_datasets