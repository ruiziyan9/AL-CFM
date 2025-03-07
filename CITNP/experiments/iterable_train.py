"""
File to train the model.
"""
import argparse
from pathlib import Path

import torch
import wandb

from CITNP.datasets.dataset_generator import InterventionDatasetGenerator
from CITNP.models.causalinferencemodel import CausalInferenceModel
from CITNP.trainer.causalinferencetrainer import CausalInferenceTrainer
from CITNP.utils.default_args import overwrite_debug_args, parse_args
from CITNP.utils.utils import set_seed


def main(args: argparse.Namespace):

    # If debug mode is on, overwrite the default arguments
    if args.debug:
        args = overwrite_debug_args(args)

    set_seed(args.seed)

    DATA_ARGS = {
        "sample_size": args.sample_size,
        "num_variables": args.num_variables,
        "function_generator": args.function_generator,
        "graph_type":  args.graph_type,
        "graph_degrees": args.graph_degrees,
        "iterations_per_epoch": args.iterations_per_epoch,
        "batch_size": args.batch_size,
    }

    train_dataset = InterventionDatasetGenerator(
        **DATA_ARGS
    )
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
        "outcome_firstnode": args.outcome_firstnode,
        "d_model": args.d_model,
        "emb_depth": args.emb_depth,
        "dim_feedforward": args.dim_feedforward,
        "nhead": args.nhead,
        "dropout": 0.0,
        "num_layers_encoder": args.num_layers_encoder,
        "num_nodes": args.num_variables,
        "device": args.device,
        "dtype": args.dtype,
    }

    model = CausalInferenceModel(
        **MODEL_ARGS
    )
    save_dir = Path("/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP/experiments/results")
    model_save_dir = save_dir / f"{args.run_name}"

    TRAINER_ARGS = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "use_wandb": not args.no_wandb,
        "lr_warmup_ratio": args.lr_warmup_ratio,
        "bfloat16": args.dtype == torch.bfloat16,
        "num_workers": 0,
        "save_dir": model_save_dir,
        "cntxt_split": [0.2, 1.0],
        "sample_size": args.sample_size,
        "log_step": args.log_step,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.no_wandb:
        run = wandb.init(
            # Set the project where this run will be logged
            project="meta_causal_inference",
            name=args.run_name,
            # Track hyperparameters and run metadata
            config={**vars(args), **TRAINER_ARGS, **MODEL_ARGS},
        )
    else:
        pass

    trainer = CausalInferenceTrainer(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
        **TRAINER_ARGS,
    )
    trainer.train()


if __name__ == "__main__":
    wandb.login(key="bc359b26d166ea6980eb0e231060bd7b8c06925e")
    parser = argparse.ArgumentParser()
    args = parse_args()
    main(args)
