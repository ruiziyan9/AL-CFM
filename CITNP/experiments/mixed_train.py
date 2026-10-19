"""
Train the mixed-context MACE-TNP model (design R1) on grouped mixed-regime data.

    python -m CITNP.experiments.mixed_train \
        --data_config CITNP/experiments/configs/smoke_d3.yaml \
        --no_wandb --device cpu --dtype float32 --epochs 3

All upstream arguments from CITNP/utils/default_args.py are accepted (model
size, optimiser, logging, --num_workers, --seed, ...). Extra arguments:

- --data_config <yaml|json>: a MixedRegimeConfig with qry_mode: grouped.
- --model_config <yaml>: optional; overrides model arguments (d_model, nhead,
  num_layers_encoder, sample_attn_mode, dim_feedforward, emb_depth,
  num_mixture_components, ...).
- --init_from_upstream <ckpt>: warm start from an upstream MACE-TNP
  checkpoint (only the new intervened-flag embedding starts from zero).
- --val_iterations / --test_iterations: fixed validation / test batches.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml  # type: ignore[import-untyped]
from torch.optim import Adam

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.epoch_callback import SetEpochCallback
from CITNP.datasets.mixed_regime_generator import (
    MixedRegimeConfig,
    MixedRegimeTaskDataset,
)
from CITNP.models.mixed_context_model import (
    MixedContextMoGModel,
    load_upstream_checkpoint,
)
from CITNP.trainer.callbacks import (
    Callback,
    CheckpointCallback,
    LRSchedulerCallback,
    ProgressBarCallback,
    WandbLoggingCallback,
)
from CITNP.trainer.mixed_context_trainer import MixedContextTrainer
from CITNP.trainer.monitor_callbacks import OracleGapCallback, TrainStatsCallback
from CITNP.utils.configs import CausalInfModelConfig
from CITNP.utils.default_args import overwrite_debug_args, parse_default_args
from CITNP.utils.utils import (
    _generate_experiment_name,
    convert_trainerdict_to_config,
    set_seed,
)


class CosineLRCallback(Callback):
    """Linear warmup over `warmup_ratio` of training, then cosine decay from
    base_lr to min_lr_ratio * base_lr. Sets the LR before every step."""

    def __init__(
        self, base_lr: float, warmup_ratio: float, min_lr_ratio: float = 0.02
    ):
        self.base_lr = base_lr
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr_ratio * base_lr

    def on_train_begin(self, trainer):
        self.total = max(1, trainer.total_train_steps)
        self.warmup = int(self.warmup_ratio * self.total)
        print(f"Cosine LR: {self.warmup} warmup steps, {self.total} total steps.")

    def on_batch_begin(self, trainer, batch_idx):
        s = trainer.global_step
        if s < self.warmup:
            lr = self.base_lr * (s + 1) / self.warmup
        else:
            frac = (s - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * min(1.0, frac))
            )
        for g in trainer.optimizer.param_groups:
            g["lr"] = lr


MODEL_KEYS = (
    "d_model",
    "emb_depth",
    "dim_feedforward",
    "nhead",
    "num_layers_encoder",
    "sample_attn_mode",
    "linear_attention",
    "mean_loss_across_samples",
    "num_mixture_components",
)


def parse_args(argv=None) -> argparse.Namespace:
    """Mixed-training arguments plus every upstream default argument."""
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--data_config", required=True)
    extra.add_argument("--model_config", default=None)
    extra.add_argument("--init_from_upstream", default=None)
    extra.add_argument("--val_iterations", type=int, default=20)
    extra.add_argument("--test_iterations", type=int, default=20)
    extra.add_argument(
        "--lr_schedule", choices=["constant", "cosine"], default="constant"
    )
    extra.add_argument(
        "--oracle_bank",
        default=None,
        help="pickle from `python -m CITNP.evaluation bank`; enables OracleGapCallback",
    )
    extra.add_argument("--oracle_every", type=int, default=2500)
    extra.add_argument(
        "--resume_from",
        default=None,
        help="checkpoint of this model to start from (weights only; fresh optimiser)",
    )
    argv = sys.argv[1:] if argv is None else argv
    mixed, rest = extra.parse_known_args(argv)
    # parse_default_args reads sys.argv itself.
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *rest]
        args = parse_default_args()
    finally:
        sys.argv = old_argv
    for k, v in vars(mixed).items():
        setattr(args, k, v)
    return args


def build_trainer(args: argparse.Namespace, data_cfg: MixedRegimeConfig):
    """Datasets, model, configs and callbacks; returns (trainer, model_config)."""
    if data_cfg.qry_mode != "grouped":
        raise ValueError("mixed_train needs a data config with qry_mode: grouped")
    if args.model_config:
        with open(args.model_config) as f:
            for k, v in yaml.safe_load(f).items():
                if k not in MODEL_KEYS:
                    raise ValueError(f"Unknown model_config key {k!r}")
                setattr(args, k, v)

    train_ds = MixedRegimeTaskDataset(
        data_cfg,
        args.batch_size,
        base_seed=args.seed,
        split="train",
        iterations_per_epoch=args.iterations_per_epoch,
    )
    # Validation / test repeat the same tasks on every pass.
    val_ds = MixedRegimeTaskDataset(
        data_cfg,
        args.batch_size,
        base_seed=args.seed,
        split="val",
        iterations_per_epoch=args.val_iterations,
        fresh_each_epoch=False,
    )
    test_ds = MixedRegimeTaskDataset(
        data_cfg,
        args.batch_size,
        base_seed=args.seed,
        split="test",
        iterations_per_epoch=args.test_iterations,
        fresh_each_epoch=False,
    )

    model_config = CausalInfModelConfig(
        d_model=args.d_model,
        emb_depth=args.emb_depth,
        dim_feedforward=args.dim_feedforward,
        nhead=args.nhead,
        dropout=0.0,
        num_layers_encoder=args.num_layers_encoder,
        num_nodes=max(data_cfg.num_variables),
        sample_attn_mode=args.sample_attn_mode,
        linear_attention=args.linear_attention,
        mean_loss_across_samples=args.mean_loss_across_samples,
        device=args.device,
        dtype=args.dtype,
        num_mixture_components=args.num_mixture_components,
    )
    model = MixedContextMoGModel(model_config)
    if args.init_from_upstream:
        load_upstream_checkpoint(model, args.init_from_upstream)
    if args.resume_from:
        state = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))  # strict
        print(f"Loaded weights from {args.resume_from}")

    results_path = (
        Path.cwd() if args.results_path is None else Path(args.results_path)
    )
    model_save_dir = results_path / "results" / f"{args.run_name}"
    model_save_dir.parent.mkdir(parents=True, exist_ok=True)
    # Don't allow to override models
    model_save_dir.mkdir(parents=False, exist_ok=False)

    dataconfig, optimizerconfig, trainingconfig, loggingconfig = (
        convert_trainerdict_to_config(
            {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "use_wandb": not args.no_wandb,
                "lr_warmup_ratio": args.lr_warmup_ratio,
                "num_workers": args.num_workers,
                "save_dir": model_save_dir,
                # Unused by the mixed trainer (batches come pre-collated).
                "cntxt_split": [0.05, 0.75],
                "sample_size": 0,
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
                # Upstream's validation plots expect upstream tuple batches.
                "plot_validation_samples": False,
                "num_validation_plots": 0,
            }
        )
    )

    callbacks = [
        WandbLoggingCallback(
            entity_name=args.entity_name,
            project_name="meta_causal_inference",
            run_name=args.run_name,
            log_config=loggingconfig,
            optim_config=optimizerconfig,
            data_config=dataconfig,
            model_config=model_config,
            train_config=trainingconfig,
        ),
        ProgressBarCallback(),
        CheckpointCallback(log_config=loggingconfig),
        (
            CosineLRCallback(args.learning_rate, args.lr_warmup_ratio)
            if args.lr_schedule == "cosine"
            else LRSchedulerCallback(optim_config=optimizerconfig)
        ),
        # Fresh training tasks every epoch (also with non-persistent workers).
        SetEpochCallback([train_ds]),
        TrainStatsCallback(
            every=args.log_step, clip_val=trainingconfig.gradient_clip_val
        ),
    ]
    if args.oracle_bank:
        callbacks.append(
            OracleGapCallback(args.oracle_bank, every=args.oracle_every)
        )

    trainer = MixedContextTrainer(
        train_dataset=train_ds,
        validation_dataset=val_ds,
        test_dataset=test_ds,
        model=model,
        dataconfig=dataconfig,
        trainingconfig=trainingconfig,
        optimizerconfig=optimizerconfig,
        loggingconfig=loggingconfig,
        callbacks=callbacks,
        seed=args.seed,
    )
    return trainer, model_config


def main(args: argparse.Namespace):
    if args.debug:
        args = overwrite_debug_args(args)
    if args.run_name is None:
        args.run_name = _generate_experiment_name()
    set_seed(args.seed)
    data_cfg = load_config(args.data_config)
    trainer, _ = build_trainer(args, data_cfg)
    trainer.train()
    test_stats = trainer.test()
    print(test_stats)
    return trainer, test_stats


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.distributions.Distribution.set_default_validate_args(False)
    main(parse_args())
