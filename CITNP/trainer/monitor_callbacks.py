"""
Training monitors for the mixed-context model.

- TrainStatsCallback: every `every` steps, the window's mean loss, mean / max
  gradient norm before clipping, fraction of clipped steps, learning rate,
  steps per second and the fraction of wall time spent waiting for data.
- OracleGapCallback: every `every` steps, the model's excess NLL over the exact
  Bayes oracle on a fixed bank (CITNP/evaluation/oracle_bank.py), overall and
  per slice, plus calibration (50% / 90% interval coverage), the effective
  number of mixture components, and the log-density gained from the
  intervened-cell flags (flags on vs all-False).

Both append rows to a CSV in the run directory (so results survive a Colab
disconnect on Drive) and log to W&B when a W&B run is active.
"""

import csv
import pickle
import time
from pathlib import Path

import numpy as np

from CITNP.evaluation.model_api import (
    effective_components,
    mixture_cdf,
    mixture_logpdf,
    predict_mixture,
)
from CITNP.trainer.callbacks import Callback

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _log_row(row: dict, path: Path, step: int) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", *row])
        if new:
            w.writeheader()
        w.writerow({"step": step, **row})
    if wandb is not None and wandb.run is not None:
        wandb.log(row, step=step)


# --------------------------------------------------------------------------- #
# Training statistics
# --------------------------------------------------------------------------- #
class TrainStatsCallback(Callback):
    """Needs MixedContextTrainer.train_step to return "grad_norm"."""

    def __init__(self, every: int, clip_val: float | None):
        self.every = every
        self.clip_val = clip_val

    def _reset(self):
        self.losses, self.norms = [], []
        self.wait = 0.0
        self.t_window = time.perf_counter()
        self.val_time = 0.0
        self.val_counted = 0.0
        self.skipped = 0.0

    def on_validation_begin(self, trainer):
        self.t_val = time.perf_counter()

    def on_validation_end(self, trainer, validation_logs):
        self.val_time += time.perf_counter() - self.t_val

    def on_train_begin(self, trainer):
        self.path = Path(trainer.loggingconfig.save_dir) / "train_stats.csv"
        self._reset()
        self.t_last_end = time.perf_counter()

    def on_batch_begin(self, trainer, batch_idx):
        # Time since the last step ended, minus validation run in between,
        # is time spent waiting for the DataLoader (plus other callbacks).
        self.wait += time.perf_counter() - self.t_last_end
        self.wait -= self.val_time - self.val_counted
        self.val_counted = self.val_time

    def on_batch_end(self, trainer, batch_idx, batch_logs):
        self.losses.append(batch_logs["loss"])
        self.norms.append(batch_logs.get("grad_norm", float("nan")))
        if trainer.global_step % self.every == 0:
            elapsed = time.perf_counter() - self.t_window
            norms = np.array(self.norms)
            row = {
                "train/loss_mean": float(np.nanmean(self.losses)),
                "train/grad_norm_mean": float(np.nanmean(norms)),
                "train/grad_norm_max": float(np.nanmax(norms)),
                "train/clip_frac": (
                    float(np.mean(norms > self.clip_val)) if self.clip_val else 0.0
                ),
                "train/lr": float(trainer.optimizer.param_groups[0]["lr"]),
                "train/steps_per_sec": len(self.losses) / elapsed,
                "train/data_wait_frac": self.wait / elapsed,
                "train/val_frac": self.val_time / elapsed,
                "train/skipped": self.skipped,
            }
            _log_row(row, self.path, trainer.global_step)
            self._reset()
        self.t_last_end = time.perf_counter()
        self.skipped += batch_logs.get("skipped", 0.0)


# --------------------------------------------------------------------------- #
# Gap to the Bayes oracle
# --------------------------------------------------------------------------- #
def _item_scores(model, item, device, normalise=True, zero_flags=False):
    """Per query point of a bank item: log p_model(y | do(T = t), context) on
    the RAW scale, the PIT value F_model(y), and exp(entropy) of the weights."""
    mix = predict_mixture(
        model, item["ctx_x"], item["ctx_int"], int(item["T"]), int(item["Y"]),
        item["t"], device=device, zero_flags=zero_flags, normalise=normalise,
    )  # fmt: skip
    logp = mixture_logpdf(*mix, item["y"])[0]
    pit = mixture_cdf(*mix, item["y"])[0]
    eff = effective_components(mix[2])[0]
    return logp.cpu().numpy(), pit.cpu().numpy(), eff.cpu().numpy()


class OracleGapCallback(Callback):
    def __init__(self, bank_path: str, every: int = 2500, flag_ablation: bool = True):
        self.bank_path = bank_path
        self.every = every
        self.flag_ablation = flag_ablation

    def on_train_begin(self, trainer):
        with open(self.bank_path, "rb") as f:
            self.items = pickle.load(f)["items"]
        self.path = Path(trainer.loggingconfig.save_dir) / "oracle_gap.csv"
        self._evaluate(trainer)  # step-0 baseline

    def on_batch_end(self, trainer, batch_idx, batch_logs):
        if trainer.global_step % self.every == 0:
            self._evaluate(trainer)

    def on_train_end(self, trainer):
        if trainer.global_step % self.every != 0:
            self._evaluate(trainer)

    def _evaluate(self, trainer):
        t0 = time.perf_counter()
        model = trainer.model
        was_training = model.training
        model.eval()
        normalise = getattr(trainer, "normalise", True)
        gaps, pits, effs, flag_gain = {}, [], [], []
        for it in self.items:
            lp, pit, eff = _item_scores(model, it, trainer.device, normalise)
            gap = float(np.mean(it["oracle_logp"] - lp))  # nats per query point
            for lab in it["labels"]:
                gaps.setdefault(lab, []).append(gap)
            pits.append(pit)
            effs.append(eff)
            if self.flag_ablation and it["n_int_T"] > 0:
                lp0, _, _ = _item_scores(
                    model, it, trainer.device, normalise, zero_flags=True
                )
                flag_gain.append(float(np.mean(lp - lp0)))
        if was_training:
            model.train()

        pit = np.concatenate(pits)
        row = {f"oracle_gap/{k}": float(np.mean(v)) for k, v in sorted(gaps.items())}
        row["calib/cover50"] = float(np.mean(np.abs(pit - 0.5) < 0.25))
        row["calib/cover90"] = float(np.mean(np.abs(pit - 0.5) < 0.45))
        row["mog/effective_components"] = float(np.mean(np.concatenate(effs)))
        if self.flag_ablation:
            row["flag/logp_gain"] = float(np.mean(flag_gain)) if flag_gain else float("nan")
        _log_row(row, self.path, trainer.global_step)
        print(
            f"[oracle] step {trainer.global_step}: gap all {row['oracle_gap/all']:.4f}"
            f" | nT=1-10 {row.get('oracle_gap/nT=1-10', float('nan')):.4f}"
            f" | cover90 {row['calib/cover90']:.2f}"
            f" ({time.perf_counter() - t0:.0f}s)"
        )
