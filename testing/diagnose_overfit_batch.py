"""Why does test_overfits_one_batch fail after commit 929f605 (n_obs_min)?

Runs the test's exact procedure (seed 0, tiny MoG model, Adam 1e-3, 200 steps)
on two fixed batches of 8 tasks:
    old  n_obs_min=0   reproduces the generator before 929f605 byte for byte
    new  n_obs_min=10  current default (what the test uses now)
and prints, per task, the design and the NLL per query row before and after
training, so we can see which tasks keep the loss high.

    PYTHONPATH=. python testing/diagnose_overfit_batch.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import D, tiny_config  # noqa: E402

from CITNP.datasets.mixed_regime_generator import (  # noqa: E402
    MixedRegimeConfig,
    descendants,
    generate_task,
)
from CITNP.models.mixed_context_model import MixedContextMoGModel  # noqa: E402

NQ = 16


def make_batch(n_obs_min: int):
    """Same config and seeds as test_mixed_context_model._real_batch."""
    cfg = MixedRegimeConfig(
        "linearfixedstd", [D], ["ER"], {D: [2, 3]}, n_ctx_range=(64, 64),
        n_qry=NQ, qry_mode="grouped", qry_groups=[1], n_obs_min=n_obs_min,
    )  # fmt: skip
    tasks = [generate_task(cfg, s, n_groups=1) for s in range(8)]
    ctx = torch.tensor(np.stack([t.ctx_x for t in tasks]))[..., None]
    cint = torch.tensor(np.stack([t.ctx_int for t in tasks]))
    tgt = torch.tensor(np.stack([t.qry_x for t in tasks]))[..., None]
    T = torch.tensor([t.meta["group_treatments"][0] for t in tasks])
    Y = (T + 1) % D
    return tasks, (ctx, cint, tgt, T, Y)


def per_task_nll(model, ctx, cint, tgt, T, Y) -> np.ndarray:
    """NLL per query row of each task (the test's loss sums rows and
    averages tasks, so loss = NQ * mean of this)."""
    out = []
    with torch.no_grad():
        for b in range(len(T)):
            s = slice(b, b + 1)
            o = model(ctx[s], tgt[s], T[s], Y[s], context_int_mask=cint[s])
            out.append(model.calculate_loss(o, target=tgt[s], outcome_index=Y[s]).item())
    return np.array(out) / NQ


def run(name: str, n_obs_min: int) -> None:
    torch.manual_seed(0)  # same order as the test: model first, then batch
    model = MixedContextMoGModel(tiny_config()).train()
    tasks, (ctx, cint, tgt, T, Y) = make_batch(n_obs_min)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn():
        out = model(ctx, tgt, T, Y, context_int_mask=cint)
        return model.calculate_loss(out, target=tgt, outcome_index=Y)

    first, nll0 = loss_fn().item(), per_task_nll(model, ctx, cint, tgt, T, Y)
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    last, nll1 = loss_fn().item(), per_task_nll(model, ctx, cint, tgt, T, Y)
    ok = first - last > 0.5 * abs(first)
    print(f"\n=== {name} (n_obs_min={n_obs_min}): first {first:.2f}  last {last:.2f}  "
          f"drop {first - last:.2f} (needs > {0.5 * abs(first):.2f}) -> "
          f"{'PASS' if ok else 'FAIL'}")  # fmt: skip
    print(f"{'seed':>4} {'has_obs':>7} {'K':>2} {'regime_sizes':>16} {'T':>2} {'Y':>2} "
          f"{'Y desc T':>8} {'do_dist':>22} {'sd(y)':>6} {'max|y|':>7} "
          f"{'nll0/row':>8} {'nll1/row':>8}")  # fmt: skip
    for b, t in enumerate(tasks):
        Tb, Yb = int(T[b]), int(Y[b])
        y = t.qry_x[:, Yb]
        dd = t.meta["do_dist"]
        dd_s = (f"normal sd={dd['scale']:.2f}" if dd["kind"] == "normal"
                else f"uniform w={dd['width']:.2f}")  # fmt: skip
        print(f"{b:>4} {str(t.meta['has_obs']):>7} {t.meta['K']:>2} "
              f"{str(t.meta['regime_sizes']):>16} {Tb:>2} {Yb:>2} "
              f"{str(Yb in descendants(t.graph, Tb)):>8} {dd_s:>22} "
              f"{y.std():6.2f} {np.abs(y).max():7.2f} {nll0[b]:8.3f} {nll1[b]:8.3f}")  # fmt: skip


if __name__ == "__main__":
    run("old batch", 0)
    run("new batch", 10)
