"""
File to train the model.
"""

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.optim import Adam

from CITNP.datasets.dataset_generator import InterventionDatasetGenerator
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

    few_nodes_graph_degrees = {
        5: np.arange(5, 10, dtype=int).tolist(),
        6: np.arange(6, 15, dtype=int).tolist(),
        7: np.arange(7, 21, dtype=int).tolist(),
        8: np.arange(8, 28, dtype=int).tolist(),
        9: np.arange(9, 36, dtype=int).tolist(),
        10: np.arange(10, 45, dtype=int).tolist(),
        11: np.arange(11, 55, dtype=int).tolist(),
        12: np.arange(12, 66, dtype=int).tolist(),
    }
    large_graph_degrees = {
        k: np.arange(k, 6 * k, dtype=int).tolist() for k in range(13, 51)
    }
    graph_degrees = {**few_nodes_graph_degrees, **large_graph_degrees}

    DATA_ARGS = {
        "sample_size": args.sample_size,
        "num_variables": list(np.arange(8, 12)),
        "function_generator": args.function_generator,
        "graph_type": ["ER", "SF"],
        "graph_degrees": graph_degrees,
        "iterations_per_epoch": args.iterations_per_epoch,
        "batch_size": args.batch_size,
        "same_variablenum_per_batch": True,
    }

    train_dataset = InterventionDatasetGenerator(**DATA_ARGS)
    DATA_ARGS.pop("iterations_per_epoch")
    val_dataset = InterventionDatasetGenerator(
        iterations_per_epoch=100 // args.batch_size,
        **DATA_ARGS,
    )
    test_dataset = InterventionDatasetGenerator(
        iterations_per_epoch=100 // args.batch_size,
        **DATA_ARGS,
    )

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
    entity_name = args.entity_name
    project_name = args.data_file

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
    project_name = "meta_causal_inference"

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
    test_stats = trainer.test()


if __name__ == "__main__":
    # These fixed some attention errors for me
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    parser = argparse.ArgumentParser()
    args = parse_default_args()
    main(args)
