"""Can the test's tiny model memorise one fixed batch? (test_overfits_one_batch)

Trains the tiny MoG model of test_mixed_context_model with the test's optimiser
(Adam, lr 1e-3, full batch, seed 0) for 3000 steps on fixed batches of 8 grouped
G=1 linear tasks (the test's generator config), and logs at 200, 500, 1000,
2000 and 3000 steps the mean NLL per query row, every task's NLL per row and
how many tasks are above their starting value.

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


def run(label: str, n_obs_min: int, seed0: int) -> None:
    t0 = time.perf_counter()
    torch.manual_seed(0)  # same order as the test: model first, then batch
    model = MixedContextMoGModel(tiny_config()).train()
    tasks, batch = make_batch(n_obs_min, seed0)
    ctx, cint, tgt, T, Y = batch
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn():
        out = model(ctx, tgt, T, Y, context_int_mask=cint)
        return model.calculate_loss(out, target=tgt, outcome_index=Y)

    nll = {0: per_task_nll(model, *batch)}
    first_nonfinite = None
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = loss_fn()
        if first_nonfinite is None and not torch.isfinite(loss):
            first_nonfinite = step
        loss.backward()
        opt.step()
        if step in CHECKPOINTS:
            nll[step] = per_task_nll(model, *batch)

    # ---- table ---------------------------------------------------------- #
    TY = [(int(T[b]), int(Y[b])) for b in range(len(tasks))]
    n_TY = Counter(TY)
    cols = [0, *CHECKPOINTS]
    print(f"\n=== {label} (n_obs_min={n_obs_min})  [{time.perf_counter() - t0:.0f}s]")
    if first_nonfinite is not None:
        print(f"!!! non-finite loss first at step {first_nonfinite}")
    head = f"{'task':>4} {'seed':>4} {'T,Y':>4} {'desc':>5} {'n(T,Y)':>6} {'sd(y)':>5} "
    head += " ".join(f"{('start' if c == 0 else f'@{c}'):>7}" for c in cols)
    print(head + "  flag")
    flagged = []
    for b, t in enumerate(tasks):
        Tb, Yb = TY[b]
        start, final = nll[0][b], nll[STEPS][b]
        flag = ""
        if final > start:
            flag = "WORSE+shared" if n_TY[(Tb, Yb)] > 1 else "WORSE"
            flagged.append((b, flag))
        print(f"{b:>4} {seed0 + b:>4} {f'{Tb},{Yb}':>4} "
              f"{str(Yb in descendants(t.graph, Tb)):>5} {n_TY[(Tb, Yb)]:>6} "
              f"{t.qry_x[:, Yb].std():5.2f} "
              + " ".join(f"{nll[c][b]:7.3f}" for c in cols)
              + f"  {flag}")  # fmt: skip
    pad = " " * 34  # width of the six leading columns
    print(f"{'mean':>4}{pad[4:]}" + " ".join(f"{nll[c].mean():7.3f}" for c in cols))
    print(f"{'#up':>4}{pad[4:]}"
          + " ".join(f"{int((nll[c] > nll[0]).sum()):>7d}" for c in cols)
          + "   (#tasks above their start)")  # fmt: skip
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
