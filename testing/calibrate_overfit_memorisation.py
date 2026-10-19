"""Can the test's tiny model memorise one fixed batch? (test_overfits_one_batch)

Trains the tiny MoG model of test_mixed_context_model with the test's optimiser
(Adam, lr 1e-3, full batch, seed 0) for 3000 steps on fixed batches of 8 grouped
G=1 linear tasks (the test's generator config), with gradient-norm clipping at
1.0 as in the trainer, and logs at 200, 500, 1000, 2000 and 3000 steps the mean
NLL per query row, every task's NLL per row, how many tasks are above their
starting value and the minimum predicted sd. A non-finite loss (or NaN/inf
mixture parameters rejected by torch.distributions) stops that batch: the step
and the minimum predicted sd are printed and the next batch runs.

Batches:
    new  n_obs_min=10 (current default), seeds 0-7, 8-15, 16-23
    old  n_obs_min=0  (generator before 929f605), seeds 0-7, for comparison

Flags at the end of each table:
    WORSE         task's final NLL is above its starting NLL
    WORSE+shared  ... and another task in the batch has the same (T, Y), so
                  the model sees the same query with a different answer

    PYTHONPATH=. python testing/calibrate_overfit_memorisation.py
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_overfit_batch import NQ, per_task_nll  # noqa: E402
from test_mixed_context_model import D, tiny_config  # noqa: E402

from CITNP.datasets.mixed_regime_generator import (  # noqa: E402
    MixedRegimeConfig,
    descendants,
    generate_task,
)
from CITNP.models.mixed_context_model import MixedContextMoGModel  # noqa: E402

CLIP = 1.0  # gradient_clip_val hard-coded in CITNP/experiments/mixed_train.py
STEPS = 3000
CHECKPOINTS = [200, 500, 1000, 2000, 3000]
RUNS = [  # (label, n_obs_min, first seed)
    ("new, seeds 0-7", 10, 0),
    ("new, seeds 8-15", 10, 8),
    ("new, seeds 16-23", 10, 16),
    ("old, seeds 0-7", 0, 0),
]


def make_batch(n_obs_min: int, seed0: int):
    """test_mixed_context_model._real_batch with seeds seed0..seed0+7."""
    cfg = MixedRegimeConfig(
        "linearfixedstd", [D], ["ER"], {D: [2, 3]}, n_ctx_range=(64, 64),
        n_qry=NQ, qry_mode="grouped", qry_groups=[1], n_obs_min=n_obs_min,
    )  # fmt: skip
    tasks = [generate_task(cfg, s, n_groups=1) for s in range(seed0, seed0 + 8)]
    ctx = torch.tensor(np.stack([t.ctx_x for t in tasks]))[..., None]
    cint = torch.tensor(np.stack([t.ctx_int for t in tasks]))
    tgt = torch.tensor(np.stack([t.qry_x for t in tasks]))[..., None]
    T = torch.tensor([t.meta["group_treatments"][0] for t in tasks])
    Y = (T + 1) % D
    return tasks, (ctx, cint, tgt, T, Y)


def _sd_stats(out) -> tuple[float, int, int]:
    """(min finite predicted sd over all tasks, query rows and components,
    #non-finite predicted means, #non-finite predicted sds)."""
    sd = out.pred_std.detach()
    mu = out.pred_mean.detach()
    finite = sd[torch.isfinite(sd)]
    min_sd = float(finite.min()) if finite.numel() else float("nan")
    return min_sd, int((~torch.isfinite(mu)).sum()), int((~torch.isfinite(sd)).sum())


def _checkpoint(model, batch):
    """Per-task NLL per row (None if it cannot be computed) and the min
    predicted sd of a full-batch forward pass, without gradients."""
    ctx, cint, tgt, T, Y = batch
    with torch.no_grad():
        min_sd = _sd_stats(model(ctx, tgt, T, Y, context_int_mask=cint))[0]
    try:
        return per_task_nll(model, *batch), min_sd
    except ValueError:  # Normal(validate_args) rejects non-finite parameters
        return None, min_sd


def run(label: str, n_obs_min: int, seed0: int) -> None:
    t0 = time.perf_counter()
    torch.manual_seed(0)  # same order as the test: model first, then batch
    model = MixedContextMoGModel(tiny_config()).train()
    tasks, batch = make_batch(n_obs_min, seed0)
    ctx, cint, tgt, T, Y = batch
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    nll, min_sd = {}, {}
    nll[0], min_sd[0] = _checkpoint(model, batch)
    failure = None
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        out, loss, error = None, None, ""
        try:
            out = model(ctx, tgt, T, Y, context_int_mask=cint)
            loss = model.calculate_loss(out, target=tgt, outcome_index=Y)
        except ValueError as e:  # Normal(validate_args) rejects NaN/inf params
            error = str(e).splitlines()[0][:120]
        if loss is None or not torch.isfinite(loss):
            sd_min, n_mu_bad, n_sd_bad = (
                _sd_stats(out) if out is not None else (float("nan"), -1, -1)
            )
            failure = dict(step=step, min_sd=sd_min, nonfinite_mu=n_mu_bad,
                           nonfinite_sd=n_sd_bad, error=error)  # fmt: skip
            break
        loss.backward()
        # As MixedContextTrainer.train_step (gradient_clip_val=1.0 in mixed_train).
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if step in CHECKPOINTS:
            nll[step], min_sd[step] = _checkpoint(model, batch)

    # ---- table ---------------------------------------------------------- #
    TY = [(int(T[b]), int(Y[b])) for b in range(len(tasks))]
    n_TY = Counter(TY)
    cols = [0, *CHECKPOINTS]
    done = [c for c in cols if nll.get(c) is not None]
    print(f"\n=== {label} (n_obs_min={n_obs_min}, clip {CLIP})  "
          f"[{time.perf_counter() - t0:.0f}s]")  # fmt: skip
    if failure is not None:
        print(f"!!! non-finite loss at step {failure['step']}: min predicted sd "
              f"{failure['min_sd']:.3g}, non-finite means {failure['nonfinite_mu']}, "
              f"non-finite sds {failure['nonfinite_sd']}"
              + (f"; {failure['error']}" if failure["error"] else "")
              + "  -> stopped this batch")  # fmt: skip

    def cell(c, b=None) -> str:
        if nll.get(c) is None:
            return f"{'n/a':>7}"
        return f"{(nll[c].mean() if b is None else nll[c][b]):7.3f}"

    head = f"{'task':>4} {'seed':>4} {'T,Y':>4} {'desc':>5} {'n(T,Y)':>6} {'sd(y)':>5} "
    head += " ".join(f"{('start' if c == 0 else f'@{c}'):>7}" for c in cols)
    print(head + "  flag")
    complete = nll.get(STEPS) is not None
    flagged = []
    for b, t in enumerate(tasks):
        Tb, Yb = TY[b]
        flag = ""
        if complete and nll[STEPS][b] > nll[0][b]:
            flag = "WORSE+shared" if n_TY[(Tb, Yb)] > 1 else "WORSE"
            flagged.append((b, flag))
        print(f"{b:>4} {seed0 + b:>4} {f'{Tb},{Yb}':>4} "
              f"{str(Yb in descendants(t.graph, Tb)):>5} {n_TY[(Tb, Yb)]:>6} "
              f"{t.qry_x[:, Yb].std():5.2f} "
              + " ".join(cell(c, b) for c in cols)
              + f"  {flag}")  # fmt: skip
    pad = " " * 34  # width of the six leading columns
    print(f"{'mean':>4}{pad[4:]}" + " ".join(cell(c) for c in cols))
    print(f"{'#up':>4}{pad[4:]}"
          + " ".join(f"{int((nll[c] > nll[0]).sum()):>7d}" if c in done
                     else f"{'n/a':>7}" for c in cols)
          + "   (#tasks above their start)")  # fmt: skip
    print(f"{'minsd':>5}{pad[5:]}"
          + " ".join(f"{min_sd[c]:7.1e}" if c in min_sd else f"{'n/a':>7}"
                     for c in cols)
          + "   (min predicted sd, any task/row/component)")  # fmt: skip
    if not complete:
        return
    drop = nll[0] - nll[STEPS]
    print(f"final drop per row: min {drop.min():.3f}, median {np.median(drop):.3f}, "
          f"mean {drop.mean():.3f};  test criterion at 3000 steps "
          f"(mean drop > half of mean start): "
          f"{'PASS' if drop.mean() > 0.5 * nll[0].mean() else 'FAIL'}")  # fmt: skip
    if flagged:
        print("flagged: " + ", ".join(f"task {b} ({f})" for b, f in flagged))


if __name__ == "__main__":
    for label, n_obs_min, seed0 in RUNS:
        run(label, n_obs_min, seed0)
