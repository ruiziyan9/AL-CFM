"""
File to train the model.
"""

import argparse
from pathlib import Path

import torch
import wandb

from CITNP.datasets.dataset_generator import InterventionDatasetGenerator
from CITNP.models.causalinferencemodel import (
    CausalInferenceModel,
    LocalLatentCausalInferenceModel,
)
from CITNP.trainer.causalinferencetrainer import CausalInferenceTrainer
from CITNP.utils.callbacks import GradientChecker
from CITNP.utils.configs import CausalInfModelConfig, LocalLatentConfig
from CITNP.utils.default_args import overwrite_debug_args, parse_args
from CITNP.utils.utils import _generate_experiment_name, set_seed


def main(args: argparse.Namespace):
    # If debug mode is on, overwrite the default arguments
    if args.debug:
        args = overwrite_debug_args(args)

    # If run name is not set, generate random name (before setting seed)
    if args.run_name is None:
        args.run_name = _generate_experiment_name()

    set_seed(args.seed)

    DATA_ARGS = {
        "sample_size": args.sample_size,
        "num_variables": args.num_variables,
        "function_generator": args.function_generator,
        "graph_type": args.graph_type,
        "graph_degrees": args.graph_degrees,
        "iterations_per_epoch": args.iterations_per_epoch,
        "batch_size": args.batch_size,
    }

    train_dataset = InterventionDatasetGenerator(**DATA_ARGS)
    DATA_ARGS.pop("iterations_per_epoch")
    val_dataset = InterventionDatasetGenerator(
        iterations_per_epoch=100,
        **DATA_ARGS,
    )
    test_dataset = InterventionDatasetGenerator(
        iterations_per_epoch=100,
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
        "num_z_samples_train": args.num_z_samples_train,
        "num_z_samples_eval": args.num_z_samples_eval,
        "decoder_depth": args.decoder_depth,
        "mean_loss_across_samples": args.mean_loss_across_samples,
        "device": args.device,
        "dtype": args.dtype,
    }

    locallatentconfig = LocalLatentConfig(**MODEL_ARGS)
    model = LocalLatentCausalInferenceModel(locallatentconfig)

    # config = CausalInfModelConfig(**MODEL_ARGS)
    # model = CausalInferenceModel(config)

    if args.results_path is None:
        results_path = Path.cwd()
    else:
        results_path = Path(args.results_path)
    save_dir = results_path / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir = save_dir / f"{args.run_name}"

    TRAINER_ARGS = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "use_wandb": not args.no_wandb,
        "lr_warmup_ratio": args.lr_warmup_ratio,
        "bfloat16": args.dtype == torch.bfloat16,
        "num_workers": 0,
        "save_dir": model_save_dir,
        "cntxt_split": [0.2, 0.8],
        "sample_size": args.sample_size,
        "log_step": args.log_step,
        "save_checkpoint_every": args.save_checkpoint_every,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.no_wandb:
        wandb.init(  # type: ignore
            # Set the entity
            entity="npcausalinf",
            # Set the project where this run will be logged
            project="zlocal",
            name=args.run_name,
            # Track hyperparameters and run metadata
            config={**vars(args), **TRAINER_ARGS, **MODEL_ARGS},
        )
    else:
        pass

    callback = [
        GradientChecker(
            model,
            watched_layers=[
                "decoder.layers.0.linear1.weight",
                "encoder.context_outcome.0.weight",
                "encoder.encoder.layers.0.linear1.weight",
                "encoder.encoder.layers.7.linear1.weight",
                "z_rep_merger.0.fc1.weight",
                "z_param_mlp.0.0.weight",
                "z_encoder.0.fc1.weight",
                "train_z_sampler.z_mean.weight",
                "train_z_sampler.z_std.weight",
            ],
        )
    ]

    trainer = CausalInferenceTrainer(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
        callback=callback,
        **TRAINER_ARGS,
    )
    trainer.train()


if __name__ == "__main__":
    # These fixed some attention errors for me
    torch.backends.cuda.enable_flash_sdp(True)

    parser = argparse.ArgumentParser()
    args = parse_args()
    main(args)
