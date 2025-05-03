"""
File to train the model.
"""

import argparse
from pathlib import Path
import json

import torch
import wandb

from typing import Union
from CITNP.models.causalinferencemodel import (
    CausalInferenceModel,
    MoGCausalInferenceModel,
    LocalLatentCausalInferenceModel,
)
from CITNP.trainer.causalinferencetrainer import CausalInferenceTester
from CITNP.utils.configs import CausalInfModelConfig, LocalLatentConfig
from CITNP.utils.datautils import MultipleFileDataset
from CITNP.utils.default_args import overwrite_debug_args, parse_args
from CITNP.utils.utils import (
    _generate_experiment_name,
    set_seed,
    load_latest_checkpoint,
)


def main(args: argparse.Namespace):
    # If debug mode is on, overwrite the default arguments
    if args.debug:
        args = overwrite_debug_args(args)

    # If run name is not set, generate random name (before setting seed)
    if args.run_name is None:
        args.run_name = _generate_experiment_name()

    set_seed(args.seed)

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "datasets/synth_training_data" / args.data_file
    # Get the training and validation datasets
    test_dir = data_dir / "test"
    test_files = list(test_dir.iterdir())
    test_dataset = MultipleFileDataset(
        [i for i in test_files if i.suffix == ".hdf5"]
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

    config: Union[CausalInfModelConfig, LocalLatentConfig]
    if args.model_type == "cnp":
        config = CausalInfModelConfig(**MODEL_ARGS)
        if args.num_mixture_components > 1:
            model = MoGCausalInferenceModel(config)
        else:
            model = CausalInferenceModel(config)
        project_name = "meta_causal_inference"
    else:
        config = LocalLatentConfig(**MODEL_ARGS)
        model = LocalLatentCausalInferenceModel(config)
        project_name = "meta_causal_inference"

    if args.results_path is None:
        results_path = Path.cwd()
    else:
        results_path = Path(args.results_path)
    save_dir = results_path / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir = save_dir / f"{args.run_name}"

    if args.model_chkpt_path == "":
        model, _ = load_latest_checkpoint(model, model_save_dir)
    else:
        model = model.load_state_dict(torch.load(args.model_chkpt_path))

    model = model.to(args.device)

    TRAINER_ARGS = {
        "batch_size": args.batch_size,
        "use_wandb": not args.no_wandb,
        "num_workers": args.num_workers,
        "sample_size": args.sample_size,
    }
    if not args.no_wandb:
        wandb.init(  # type: ignore
            # Set the entity
            entity="npcausalinf",
            # Set the project where this run will be logged
            project=project_name,
            name=args.run_name,
            resume="allow",
            # Track hyperparameters and run metadata
            config={**vars(args), **TRAINER_ARGS, **MODEL_ARGS},
        )
    else:
        pass

    tester = CausalInferenceTester(
        model=model,
        test_dataset=test_dataset,
        **TRAINER_ARGS,
    )
    test_metrics = tester.test()
    print(f"Test metrics: {test_metrics}")
    with open(model_save_dir / f"test_metrics_{args.data_file}.json", "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    # These fixed some attention errors for me
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)

    parser = argparse.ArgumentParser()
    args = parse_args()
    main(args)
