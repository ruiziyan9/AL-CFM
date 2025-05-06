"""
File to train the model.
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch.optim import Adam

from CITNP.models.causalinferencemodel import (
    CausalInferenceModel,
    LocalLatentCausalInferenceModel,
    MoGCausalInferenceModel,
)
from CITNP.trainer.callbacks import (
    CheckpointCallback,
    LRSchedulerCallback,
    ProgressBarCallback,
    WandbLoggingCallback,
)
from CITNP.trainer.causalinferencetrainer import CausalInferenceTrainer
from CITNP.utils.configs import CausalInfModelConfig, LocalLatentConfig
from CITNP.utils.datautils import ChunkMultipleFileDataset, MultipleFileDataset
from CITNP.utils.default_args import overwrite_debug_args, parse_default_args
from CITNP.utils.utils import (
    _generate_experiment_name,
    convert_trainerdict_to_config,
    set_seed,
)


def main(args: argparse.Namespace):
    # If debug mode is on, overwrite the default arguments
    if args.debug:
        args = overwrite_debug_args(args)

    # If run name is not set, generate random name (before setting seed)
    if args.run_name is None:
        args.run_name = _generate_experiment_name()

    set_seed(args.seed)

    dataset_obj = (
        partial(ChunkMultipleFileDataset, batch_size=args.batch_size)
        if "alldata" in args.data_file
        else MultipleFileDataset
    )

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "datasets/synth_training_data" / args.data_file
    # Get the training and validation datasets
    train_dir = data_dir / "train"
    train_files = list(train_dir.iterdir())
    train_hdf5_files = [i for i in train_files if i.suffix == ".hdf5"]
    train_hdf5_files = [train_hdf5_files[0]] if args.debug else train_hdf5_files
    train_dataset = dataset_obj(train_hdf5_files)

    val_dir = data_dir / "val"
    val_files = list(val_dir.iterdir())
    val_dataset = dataset_obj([i for i in val_files if i.suffix == ".hdf5"])

    test_dir = data_dir / "test"
    test_files = list(test_dir.iterdir())
    test_dataset = dataset_obj([i for i in test_files if i.suffix == ".hdf5"])

    MODEL_ARGS = {
        "d_model": args.d_model,
        "emb_depth": args.emb_depth,
        "dim_feedforward": args.dim_feedforward,
        "nhead": args.nhead,
        "dropout": 0.0,
        "num_layers_encoder": args.num_layers_encoder,
        "num_nodes": args.num_variables,
        "sample_attn_mode": args.sample_attn_mode,
        "linear_attention": args.linear_attention,
        "mean_loss_across_samples": args.mean_loss_across_samples,
        "device": args.device,
        "dtype": args.dtype,
    }

    # Add model-specific args dynamically
    if args.model_type == "cnp":
        MODEL_ARGS["num_mixture_components"] = args.num_mixture_components
    elif args.model_type == "locallatent":
        MODEL_ARGS.update(
            {
                "num_z_samples_train": args.num_z_samples_train,
                "num_z_samples_eval": args.num_z_samples_eval,
                "decoder_depth": args.decoder_depth,
            }
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if args.results_path is None:
        results_path = Path.cwd()
    else:
        results_path = Path(args.results_path)
    save_dir = results_path / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir = save_dir / f"{args.run_name}"
    # Don't allow to override models
    model_save_dir.mkdir(parents=False, exist_ok=False)

    # All the configs
    config: Union[CausalInfModelConfig, LocalLatentConfig]
    if args.model_type == "cnp":
        config = CausalInfModelConfig(**MODEL_ARGS)
        if args.num_mixture_components > 1:
            model = MoGCausalInferenceModel(config)
        else:
            model = CausalInferenceModel(config)
    else:
        config = LocalLatentConfig(**MODEL_ARGS)
        model = LocalLatentCausalInferenceModel(config)
    # project_name = "meta_causal_inference"
    project_name = args.data_file
    entity_name = args.entity_name

    TRAINER_CONFIG = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "use_wandb": not args.no_wandb,
        "lr_warmup_ratio": args.lr_warmup_ratio,
        "num_workers": args.num_workers,
        "save_dir": model_save_dir,
        "cntxt_split": [0.05, 0.75],
        "sample_size": args.sample_size,
        "log_step": args.log_step,
        "save_checkpoint_every_n_steps": args.save_checkpoint_every_n_steps,
        "normalise": args.normalise,
        "train_dtype": args.dtype,
        "eval_dtype": "float32",
        "pin_memory": True,
        "learning_rate": args.learning_rate,
        "optimizer": Adam(model.parameters(), lr=args.learning_rate),
        "gradient_clip_val": 1.0,
        "device": args.device,
        "plot_validation_samples": True,
        "num_validation_plots": 10,
    }
    dataconfig, optimizerconfig, trainingconfig, loggingconfig = (
        convert_trainerdict_to_config(TRAINER_CONFIG)
    )

    # Callbacks
    wandbcallback = WandbLoggingCallback(
        entity_name=entity_name,
        project_name=project_name,
        run_name=args.run_name,
        log_config=loggingconfig,
        optim_config=optimizerconfig,
        data_config=dataconfig,
        model_config=config,
        train_config=trainingconfig,
    )
    progress_callback = ProgressBarCallback()
    checkpoint_callback = CheckpointCallback(
        log_config=loggingconfig,
    )
    lr_scheduler_callback = LRSchedulerCallback(
        optim_config=optimizerconfig,
    )
    callbacks = [
        wandbcallback,
        progress_callback,
        checkpoint_callback,
        lr_scheduler_callback,
    ]

    trainer = CausalInferenceTrainer(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        model=model,
        dataconfig=dataconfig,
        trainingconfig=trainingconfig,
        optimizerconfig=optimizerconfig,
        loggingconfig=loggingconfig,
        callbacks=callbacks,
    )
    trainer.train()
    # Currently we aren't running testing on the entire test set so will comment out for now
    # test_stats = trainer.test()


if __name__ == "__main__":
    # These fixed some attention errors for me
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    parser = argparse.ArgumentParser()
    args = parse_default_args()
    main(args)
