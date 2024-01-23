# Optional sparse fine-tuning for BReg
    sparse_fine_tune_time = Time.from_timestring(args.sparse_fine_tune)
    if args.pruner == "BReg" and sparse_fine_tune_time.value > 0:
        final_ratio_mask = BReg.final_ratio_mask
        sparse_fine_tune = Sparse_Finetune(
            mask=final_ratio_mask,
            non_mask_name=args.non_mask_name, 
            clipping_threshold=args.clipping_threshold
        )
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
        # initialize the trainer
        trainer = Trainer(
            # training
            model=composer_model,
            train_dataloader=train_dataloader,
            optimizers=optimizer,
            max_duration=args.sparse_fine_tune,
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

            # algorithms. Gradient clipping is implemented inside the BReg pruner
            algorithms=[sparse_fine_tune],

            # checkpointing
            run_name=args.run_name+"-sparse-fine-tune",
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