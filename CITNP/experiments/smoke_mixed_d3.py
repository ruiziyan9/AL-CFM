"""
Smoke test for mixed-context training (R1) on D=3 linear-Gaussian tasks.

    python -m CITNP.experiments.smoke_mixed_d3 [--device cuda] [--num_workers 2]

Trains MixedContextMoGModel for 3 epochs x 50 iterations (batch 16) on
configs/smoke_d3.yaml and checks:
  1. no NaN / inf losses;
  2. training loss at the end of epoch 3 < at the start of epoch 1
     (means over 10 steps, to smooth step noise);
  3. validation loss is printed per epoch and decreases overall;
  4. uses-interventions check (printed, not enforced): on 200 validation tasks
     with an observational regime and K >= 1, loss with only the
     observational context rows minus loss with the full context. Expected
     >= 0 on seen groups.
Exits with status 1 if criterion 1, 2 or 3 fails.
"""

import argparse
import dataclasses
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_CHUNK,
    SPLIT_IDS,
    collate_tasks,
    generate_task,
    make_key,
)
from CITNP.experiments.mixed_train import build_trainer, parse_args
from CITNP.trainer.callbacks import Callback
from CITNP.trainer.mixed_context_trainer import prepare_mixed_batch
from CITNP.utils.utils import set_seed

HERE = Path(__file__).resolve().parent


class Recorder(Callback):
    """Records training losses per epoch and every validation loss."""

    def __init__(self):
        self.epoch = 0
        self.train: dict[int, list[float]] = {}
        self.val: list[float] = []

    def on_epoch_begin(self, trainer, epoch):
        self.epoch = epoch

    def on_batch_end(self, trainer, batch_idx, batch_logs):
        self.train.setdefault(self.epoch, []).append(batch_logs["loss"])

    def on_validation_end(self, trainer, validation_logs):
        self.val.append(validation_logs["val_loss"])
        print(
            f"epoch {self.epoch + 1}: val_loss = {validation_logs['val_loss']:.4f}"
        )


def task_losses(model, task, device) -> np.ndarray:
    """Per-group test NLL of one task, in the data's original scale.

    prepare_mixed_batch standardises each task with its own context
    statistics, so NLLs of different contexts live in different spaces. Adding
    log(std_Y) per query row maps them back to the same (generated) scale.
    """
    prep = prepare_mixed_batch(
        collate_tasks([task]), torch.float32, False, device=device, normalise=True
    )
    with torch.no_grad():
        out = model(
            context_data=prep["context"],
            target_data=prep["target"],
            intervention_index=prep["intervention_index"],
            outcome_index=prep["outcome_index"],
            context_int_mask=prep["context_int_mask"],
        )
        loss = model.calculate_loss(
            out,
            target=prep["target"],
            outcome_index=prep["outcome_index"],
            test=True,
        )
    _, std = prep["norm"]
    n_rows = 1 if model.mean_loss_across_samples else prep["target"].shape[1]
    log_std_y = torch.log(std[0, prep["outcome_index"].cpu()])
    return (loss.cpu() + n_rows * log_std_y).numpy()


def obs_only(task):
    """The task with only its observational context rows."""
    obs_r = task.regime_targets.index(())
    keep = task.ctx_regime == obs_r
    return dataclasses.replace(
        task,
        ctx_x=task.ctx_x[keep],
        ctx_int=task.ctx_int[keep],
        ctx_regime=task.ctx_regime[keep],
    )


def uses_interventions_check(model, data_cfg, device, seed, n_tasks=200):
    model.eval()
    diffs, seen = [], []
    i = done = 0
    while done < n_tasks:
        key = make_key(SOURCE_CHUNK, SPLIT_IDS["val"], seed, i)
        i += 1
        task = generate_task(data_cfg, key)
        if not task.meta["has_obs"] or task.meta["K"] < 1:
            continue
        full = task_losses(model, task, device)
        obs = task_losses(model, obs_only(task), device)
        done += 1
        for g, kind in enumerate(task.meta["group_types"]):
            diffs.append(obs[g] - full[g])
            seen.append(kind == "seen")
    diffs_np, seen_np = np.array(diffs), np.array(seen)
    print(
        f"uses-interventions on {n_tasks} tasks (obs-only minus full context "
        f"NLL per group, original scale): overall {diffs_np.mean():+.4f} "
        f"(n={len(diffs_np)} groups), "
        f"seen groups {diffs_np[seen_np].mean():+.4f} (n={seen_np.sum()}), "
        f"unseen groups {diffs_np[~seen_np].mean():+.4f} (n={(~seen_np).sum()})"
    )
    return diffs_np, seen_np


def main():
    cli = argparse.ArgumentParser(description="Mixed-context D=3 smoke test")
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--num_workers", type=int, default=2)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--results_path", default=None)
    opts = cli.parse_args()

    results_path = opts.results_path or tempfile.mkdtemp(prefix="smoke_mixed_d3_")
    args = parse_args(
        [
            "--data_config", str(HERE / "configs" / "smoke_d3.yaml"),
            "--model_config", str(HERE / "configs" / "smoke_d3_model.yaml"),
            "--batch_size", "16",
            "--iterations_per_epoch", "50",
            "--epochs", "3",
            "--learning_rate", "1e-3",
            "--lr_warmup_ratio", "0",
            "--num_workers", str(opts.num_workers),
            "--device", opts.device,
            "--dtype", "float32",
            "--seed", str(opts.seed),
            "--no_wandb",
            "--log_step", "1000000",  # validate only at the end of each epoch
            "--save_checkpoint_every_n_steps", "1000000",
            "--results_path", results_path,
            "--run_name", f"smoke_{int(time.time())}",
            "--val_iterations", "10",
            "--test_iterations", "10",
        ]
    )  # fmt: skip
    set_seed(args.seed)
    data_cfg = load_config(args.data_config)
    trainer, _ = build_trainer(args, data_cfg)
    rec = Recorder()
    trainer.callbacks.append(rec)
    trainer.train()

    all_train = [v for e in sorted(rec.train) for v in rec.train[e]]
    start = float(np.mean(rec.train[0][:10]))
    end = float(np.mean(rec.train[max(rec.train)][-10:]))
    ok1 = bool(np.all(np.isfinite(all_train)) and np.all(np.isfinite(rec.val)))
    ok2 = end < start
    ok3 = len(rec.val) == args.epochs and rec.val[-1] < rec.val[0]
    print(
        f"[{'PASS' if ok1 else 'FAIL'}] 1. no NaN/inf losses ({len(all_train)} train steps)"
    )
    print(
        f"[{'PASS' if ok2 else 'FAIL'}] 2. train loss: start of epoch 1 {start:.4f} -> end of epoch 3 {end:.4f}"
    )
    print(
        f"[{'PASS' if ok3 else 'FAIL'}] 3. val loss per epoch {[round(v, 4) for v in rec.val]}"
    )

    uses_interventions_check(trainer.model, data_cfg, trainer.device, args.seed)
    print(f"Checkpoints in {Path(results_path) / 'results' / args.run_name}")
    sys.exit(0 if ok1 and ok2 and ok3 else 1)


if __name__ == "__main__":
    main()
