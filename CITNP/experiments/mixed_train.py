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
- --intervention_encoding {r1,layerwise,extra_node}: how intervention
  information enters the model (default r1). --model_class <name> is still
  accepted (legacy).
- --scale_conditioning {none,stats}: stats feeds each variable's standardisation
  (log std, mean / std) to the model (information parity with the oracle;
  default none: no new parameters).
- --fine_tune_from <ckpt>: start a new run from these weights (fresh optimiser,
  step 0); not for continuing a run, see --resume_full.
- --resume_full <last_model.pt>: continue this run (model, optimiser, step, epoch)
  in its own results directory; same configs and --epochs required.
- --init_from_upstream <ckpt>: warm start from an upstream MACE-TNP
  checkpoint (only the new intervened-flag embedding starts from zero).
- --val_iterations / --test_iterations: fixed validation / test batches.
- --dataset_config <yaml> [--data_root DIR]: train from pre-simulated chunk files
  (see CITNP/datasets/dataset_config.py) instead of generating tasks on the fly.
  The model receives the same inputs; --data_config must be the dataset's prior.
  The data root defaults to the environment variable ALCFM_DATA_ROOT.
"""

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch
import yaml  # type: ignore[import-untyped]
from torch.optim import Adam

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.dataset_config import (
    PresimulatedDatasetConfig,
    file_fingerprint,
    load_presimulated_dataset_config,
    prior_config_path,
    resolve_data_root,
    split_directory,
)
from CITNP.datasets.epoch_callback import SetEpochCallback
from CITNP.datasets.mixed_regime_generator import (
    MixedRegimeChunkDataset,
    MixedRegimeConfig,
    MixedRegimeTaskDataset,
    assert_disjoint_splits,
)

from CITNP.models.mixed_context_model import (
    INTERVENTION_ENCODINGS,
    LayerwiseMixedContextModel,
    LayerwiseMixedContextMoGModel,
    load_r1_checkpoint,
    load_upstream_checkpoint,
    resolve_model_class,
)
from CITNP.models.scale_conditioning import SCALE_CONDITIONING_OPTIONS

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
    "model_class",
    "intervention_encoding",
    "scale_conditioning",
)

# Keys that select the method: if given both on the command line and in the
# model YAML, they must agree.
SELECTION_KEYS = ("model_class", "intervention_encoding", "scale_conditioning")


def parse_args(argv=None) -> argparse.Namespace:
    """Mixed-training arguments plus every upstream default argument."""
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--data_config", required=True)
    extra.add_argument("--model_config", default=None)
    extra.add_argument("--init_from_upstream", default=None)
    extra.add_argument(
        "--init_from_r1",
        default=None,
        help="R1 checkpoint to warm-start a Layerwise* model from",
    )

    extra.add_argument(
        "--model_class",
        default=None,
        help="legacy: class name in CITNP.models.mixed_context_model; "
        "prefer --intervention_encoding",
    )
    extra.add_argument(
        "--intervention_encoding",
        default=None,
        choices=sorted(INTERVENTION_ENCODINGS),
        help="how intervention information enters the model (default: r1)",
    )

    extra.add_argument(
        "--scale_conditioning",
        default=None,
        choices=list(SCALE_CONDITIONING_OPTIONS),
        help="feed the standardisation (log std, mean / std) to the model "
        "(default: none)",
    )
    extra.add_argument(
        "--dataset_config",
        default=None,
        help="pre-simulated dataset config (CITNP/experiments/configs/datasets/"
        "*.yaml): train from its chunk files instead of generating on the fly",
    )
    extra.add_argument(
        "--data_root",
        default=None,
        help="directory of simulated datasets (default: environment variable "
        "ALCFM_DATA_ROOT)",
    )
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
        "--resume_full",
        default=None,
        help="checkpoint (e.g. last_model.pt) of THIS run to continue: model, "
        "optimiser, step and epoch are restored and training goes on in the "
        "same run directory. Needs an epoch-boundary checkpoint and the same "
        "configs and --epochs",
    )
    extra.add_argument(
        "--fine_tune_from",
        default=None,
        help="start a NEW run from the weights of this checkpoint (strict; fresh "
        "optimiser, step 0, new schedule). To continue an interrupted run use "
        "--resume_full",
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


def apply_model_config(args: argparse.Namespace) -> None:
    """Copy the --model_config YAML onto args (YAML values override the
    command line, as before), except that a selection key given in both
    places must have the same value."""
    if not args.model_config:
        return
    with open(args.model_config) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        if k not in MODEL_KEYS:
            raise ValueError(f"Unknown model_config key {k!r}")
        current = getattr(args, k, None)
        if k in SELECTION_KEYS and current is not None and current != v:
            raise ValueError(
                f"{k}={current!r} on the command line contradicts "
                f"{k}={v!r} in {args.model_config}"
            )
        setattr(args, k, v)


def restore_full_state(trainer, path) -> None:
    """Continue a run from the checkpoint `path` (saved by CheckpointCallback):
    strict model weights, optimiser and scheduler state, global step, epoch and
    best validation loss. Only epoch-boundary checkpoints (last_model.pt) can be
    resumed: a mid-epoch step_N.pt would repeat part of an epoch."""
    state = torch.load(path, map_location="cpu")
    step = int(state["global_step"])
    if step % trainer.steps_per_epoch:
        raise ValueError(
            f"{path} was saved mid-epoch (step {step}, {trainer.steps_per_epoch} "
            "steps per epoch); resume from an epoch-boundary checkpoint such as "
            "last_model.pt"
        )
    trainer.model.load_state_dict(state["model_state_dict"])  # strict
    # Optimiser state is cast to the parameters' device at load time, so move
    # the model first.
    trainer.model.to(trainer.device)
    trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
    if trainer.scheduler is not None and state.get("scheduler_state_dict"):
        trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
    trainer.global_step = step
    trainer.current_epoch = step // trainer.steps_per_epoch
    for cb in trainer.callbacks:
        if isinstance(cb, CheckpointCallback) and "best_val_loss" in state:
            cb.best_val_loss = float(state["best_val_loss"])
    print(f"Resumed from {path}: step {step}, continuing at epoch {trainer.current_epoch}")


def streaming_evaluation_dataset(
    args: argparse.Namespace, data_cfg: MixedRegimeConfig, split: str
) -> MixedRegimeTaskDataset:
    """Validation ("val") or test dataset generated on the fly."""
    iterations = args.val_iterations if split == "val" else args.test_iterations
    return MixedRegimeTaskDataset(
        data_cfg,
        args.batch_size,
        base_seed=args.seed,
        split=split,
        iterations_per_epoch=iterations,
        fresh_each_epoch=False,
    )


def _json_round_trip(value):
    return json.loads(json.dumps(value))


def check_presimulated_data(
    args: argparse.Namespace,
    data_cfg: MixedRegimeConfig,
    dataset_config: PresimulatedDatasetConfig,
) -> dict[str, Path]:
    """Directories of the dataset's splits that exist on disk, keyed by split
    name ("train", "validation", "test"). Raises if the training split is
    missing, if any directory was written for another prior than `data_cfg`,
    if two directories share tasks, or if the shape groups do not split into
    whole batches."""
    if "train" not in dataset_config.splits:
        raise ValueError(f"{args.dataset_config} has no train split")
    # --data_config and the dataset's prior must be the same prior: the loaded
    # configs are compared, so comments and formatting do not matter.
    prior = load_config(prior_config_path(dataset_config))
    if dataclasses.asdict(prior) != dataclasses.asdict(data_cfg):
        raise ValueError(
            f"--data_config {args.data_config} differs from the prior "
            f"{dataset_config.prior_config} of {args.dataset_config}"
        )
    if dataset_config.tasks_per_shape_group % args.batch_size:
        raise ValueError(
            f"tasks_per_shape_group={dataset_config.tasks_per_shape_group} is "
            f"not a multiple of --batch_size={args.batch_size}"
        )
    data_root = resolve_data_root(args.data_root)
    expected_prior = _json_round_trip(dataclasses.asdict(data_cfg))
    directories: dict[str, Path] = {}
    for split in dataset_config.splits:
        directory = split_directory(dataset_config, data_root, split)
        stored_path = directory / "config.json"
        if not stored_path.is_file():
            if split == "train":
                raise FileNotFoundError(
                    f"No simulated training data: {stored_path} is missing "
                    "(generate it with generate_presimulated_data.sh)"
                )
            continue  # evaluation split not generated: falls back to streaming
        with open(stored_path) as f:
            stored = json.load(f)
        if stored["config"] != expected_prior:
            raise ValueError(
                f"{directory} was generated from a different prior than "
                f"--data_config {args.data_config}"
            )
        directories[split] = directory
    assert_disjoint_splits(*(str(d) for d in directories.values()))
    return directories


def build_chunk_datasets(
    args: argparse.Namespace,
    data_cfg: MixedRegimeConfig,
    dataset_config: PresimulatedDatasetConfig,
    directories: dict[str, Path],
):
    """(train, validation, test) datasets: chunk files where the split was
    generated, otherwise (validation / test only) the streaming dataset."""
    train_ds = MixedRegimeChunkDataset(
        str(directories["train"]),
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        iterations_per_epoch=args.iterations_per_epoch,
        num_workers=args.num_workers,
    )
    unique_tasks = dataset_config.splits["train"].num_tasks
    consumed = args.epochs * args.iterations_per_epoch * args.batch_size
    print(
        f"train: chunk files in {directories['train']}, "
        f"{args.iterations_per_epoch} batches per epoch"
    )
    print(
        f"Unique training tasks: {unique_tasks}; tasks this run consumes "
        f"(epochs x iterations_per_epoch x batch_size): {consumed}; "
        f"implied passes over the data: {consumed / unique_tasks:.2f}"
    )
    evaluation = {}
    for split, generator_split in (("validation", "val"), ("test", "test")):
        if split in directories:
            ds = MixedRegimeChunkDataset(
                str(directories[split]),
                args.batch_size,
                shuffle=False,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            print(
                f"{split}: chunk files in {directories[split]}, {len(ds)} "
                "batches (--val_iterations / --test_iterations are ignored)"
            )
        else:
            ds = streaming_evaluation_dataset(args, data_cfg, generator_split)
            print(f"{split}: generated on the fly, {len(ds)} batches")
        evaluation[split] = ds
    return train_ds, evaluation["validation"], evaluation["test"]


def record_presimulated_data(
    args: argparse.Namespace, directories: dict[str, Path], model_save_dir: Path
) -> None:
    """Write dataset_config, split directories and the fingerprints of their
    config.json files next to resolved_model_config.yaml."""
    record = {
        "dataset_config": str(args.dataset_config),
        "dataset_config_fingerprint": file_fingerprint(args.dataset_config),
        "splits": {
            split: {
                "directory": str(directory),
                "config_json_fingerprint": file_fingerprint(directory / "config.json"),
            }
            for split, directory in directories.items()
        },
    }
    with open(model_save_dir / "presimulated_data.yaml", "w") as f:
        yaml.safe_dump(record, f, sort_keys=False)


def build_trainer(args: argparse.Namespace, data_cfg: MixedRegimeConfig):
    """Datasets, model, configs and callbacks; returns (trainer, model_config)."""
    if data_cfg.qry_mode != "grouped":
        raise ValueError("mixed_train needs a data config with qry_mode: grouped")
    apply_model_config(args)

    chunk_directories: dict[str, Path] = {}
    if args.dataset_config:
        dataset_config = load_presimulated_dataset_config(args.dataset_config)
        chunk_directories = check_presimulated_data(args, data_cfg, dataset_config)
        train_ds, val_ds, test_ds = build_chunk_datasets(
            args, data_cfg, dataset_config, chunk_directories
        )
    else:
        train_ds = MixedRegimeTaskDataset(
            data_cfg,
            args.batch_size,
            base_seed=args.seed,
            split="train",
            iterations_per_epoch=args.iterations_per_epoch,
        )
        # Validation / test repeat the same tasks on every pass.
        val_ds, test_ds = (
            streaming_evaluation_dataset(args, data_cfg, split)
            for split in ("val", "test")
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
        scale_conditioning=args.scale_conditioning or "none",
    )
    model_cls, encoding = resolve_model_class(
        args.intervention_encoding, args.model_class
    )
    if args.init_from_r1 and not issubclass(
        model_cls, (LayerwiseMixedContextModel, LayerwiseMixedContextMoGModel)
    ):
        raise ValueError("--init_from_r1 only applies to the layerwise encoding")
    model = model_cls(model_config)
    print(f"Intervention encoding: {encoding} (model class {model_cls.__name__})")
    if args.init_from_upstream:
        load_upstream_checkpoint(model, args.init_from_upstream)
    if args.init_from_r1:
        load_r1_checkpoint(model, args.init_from_r1)
        print(f"Warm-started from R1 checkpoint {args.init_from_r1}")
    if args.fine_tune_from and args.resume_full:
        raise ValueError("--fine_tune_from (weights only) and --resume_full are exclusive")
    if args.resume_full and not Path(args.resume_full).is_file():
        raise FileNotFoundError(args.resume_full)
    if args.fine_tune_from:
        state = torch.load(args.fine_tune_from, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))  # strict
        print(f"Loaded weights from {args.fine_tune_from}")

    results_path = (
        Path.cwd() if args.results_path is None else Path(args.results_path)
    )
    model_save_dir = results_path / "results" / f"{args.run_name}"
    model_save_dir.parent.mkdir(parents=True, exist_ok=True)
    # Don't allow to override models; only --resume_full continues in place.
    resuming_here = bool(args.resume_full) and (
        Path(args.resume_full).resolve().parent == model_save_dir.resolve()
    )
    model_save_dir.mkdir(parents=False, exist_ok=resuming_here)

    resolved = {
        k: getattr(args, k)
        for k in MODEL_KEYS
        if k not in SELECTION_KEYS and getattr(args, k, None) is not None
    }
    resolved["model_class"] = model_cls.__name__
    resolved["scale_conditioning"] = args.scale_conditioning or "none"
    if encoding in INTERVENTION_ENCODINGS:
        resolved["intervention_encoding"] = encoding
    with open(model_save_dir / "resolved_model_config.yaml", "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    if args.dataset_config:
        record_presimulated_data(args, chunk_directories, model_save_dir)

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
    if args.resume_full:
        restore_full_state(trainer, args.resume_full)
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
