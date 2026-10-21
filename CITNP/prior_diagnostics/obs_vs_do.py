"""
Is the training prior reasonable? For linear-Gaussian priors, compares the
observational conditional p(Y | T=x) with the interventional p(Y | do(T=x))
for every training question (treatment T, outcome Y), exactly, and asks how
useful the context's experiments are for telling them apart.

One column per question class (stats.QUESTION_CLASSES):
  differ:     reverse, confounded effect, pure confounding
  identical:  unconfounded effect, unrelated
("differ" = open back-door path from T to Y.)

Rows:
  1. the example question at the class's weighted median W2 gap 
    (--example first: the first one collected): exact 90% bands of p(Y | T=x) (blue) and
     p(Y | do(T=x)) (red), with the task's observational and do(T) rows;
  2. distribution gap: W2 distance / sd(Y), averaged over x ~ observational T;
  3. context composition: share of rows by role (stats.CONTEXT_ROLES);
  4. usefulness of the do(T) rows (differ classes): their evidence against the
     observational answer, sum over do(T) rows of
     KL(p(Y | do(T=x_r)) || p(Y | T=x_r)) in nats (expected log Bayes factor;
     >= 3 nats is strong). 0 by construction in the identical classes.

    python -m CITNP.prior_diagnostics obs-vs-do \
        --config CITNP/experiments/configs/oracle_d3.yaml \
        --num_tasks 2000 --out obs_vs_do.pdf [--png] [--num_variables 3]

Exact for the linearnonidentifiable family (linear_gaussian.py). Questions are
weighted like the trainer: each task's questions equally, outcome uniform over
the other D-1 nodes. Writes <stem>.csv with one row per (task, T, Y).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_ADHOC,
    MixedRegimeConfig,
    Task,
    generate_task,
    make_key,
)
from CITNP.prior_diagnostics.linear_gaussian import (
    LinearGaussianSCM,
    gauss_kl,
    scm_from_debug,
)
from CITNP.prior_diagnostics.stats import (
    CONTEXT_ROLES,
    DIFFER_CLASSES,
    QUESTION_CLASSES,
    context_role,
    query_treatments,
    question_class,
)

__all__ = ["collect_obs_vs_do", "plot_obs_vs_do", "select_example", "question_record", "summary"]

ROLE_COLORS = ("#BDBDBD", "#C44E52", "#8172B3", "#55A868", "#DD8452", "#E5E5E5")
COND_COLOR, DO_COLOR = "#4C72B0", "#C44E52"
STRONG_NATS = 3.0
Z90 = 1.6449

EXAMPLE_MODES = ("median", "first")


def select_example(recs: list[dict], cls: str, how: str = "median") -> dict | None:
    """The record shown in row 1 for class `cls`: the question at the
    weighted median of w2_sd (an actual question, not an interpolated value),
    or the first one collected (how="first")."""
    R = [r for r in recs if r["cls"] == cls]
    if not R:
        return None
    if how == "first":
        return R[0]
    if how != "median":
        raise ValueError(f"how must be one of {EXAMPLE_MODES}, got {how!r}")
    order = np.argsort([r["w2_sd"] for r in R], kind="stable")
    cum = np.cumsum([R[i]["weight"] for i in order])
    return R[order[np.searchsorted(cum, 0.5 * cum[-1])]]

def question_record(task: Task, scm: LinearGaussianSCM, t: int, y: int) -> dict:
    """Exact gap between Y | T and Y | do(T) for one question, and what the
    task's context offers for it."""
    mu, S = scm.moments()
    ac, bc, vc = scm.conditional_line(t, y)
    ad, bd, vd = scm.interventional_line(t, y)
    # squared W2 between the Gaussians at x, averaged over x ~ N(mu_t, S_tt)
    gap_at_mean = (ad + bd * mu[t]) - (ac + bc * mu[t])
    w2 = np.sqrt(gap_at_mean**2 + (bd - bc) ** 2 * S[t, t] + (np.sqrt(vd) - np.sqrt(vc)) ** 2)

    n = dict.fromkeys(CONTEXT_ROLES, 0)
    for r, targets in enumerate(task.regime_targets):
        c = targets[0] if targets else None  # single-node regimes
        n[context_role(task.graph, t, y, c)] += int((task.ctx_regime == r).sum())
    regs_t = [r for r, tg in enumerate(task.regime_targets) if tuple(tg) == (t,)]
    x_t = task.ctx_x[np.isin(task.ctx_regime, regs_t), t].astype(np.float64)
    evidence = float(gauss_kl(ad + bd * x_t, vd, ac + bc * x_t, vc).sum())
    return {
        "D": len(task.graph), "cls": question_class(task.graph, t, y), "t": t, "y": y,
        "w2_sd": float(w2 / np.sqrt(S[y, y])),
        "slope_cond": float(bc), "slope_do": float(bd),
        "sd_ratio": float(np.sqrt(vd / vc)),
        **{f"n_{k}": v for k, v in n.items()},
        "evidence_nats": evidence,
    }  # fmt: skip


def _task_and_scm(cfg, i, base_seed, num_variables):
    key = make_key(SOURCE_ADHOC, 0, base_seed, i)
    task, debug = generate_task(cfg, key, num_variables=num_variables, return_debug=True)
    return task, scm_from_debug(debug)


def collect_obs_vs_do(
    cfg: MixedRegimeConfig,
    num_tasks: int,
    base_seed: int = 0,
    num_variables: int | None = None,
    example: str = "median",
):
    """Records for every training question of tasks 0..num_tasks-1, and one
    example (task, scm, t, y, rec) per class chosen by select_example."""
    recs = []
    for i in trange(num_tasks, desc="tasks"):
        task, scm = _task_and_scm(cfg, i, base_seed, num_variables)
        D = len(task.graph)
        ts = query_treatments(task)
        for t in ts:
            for y in (v for v in range(D) if v != t):
                rec = question_record(task, scm, t, y)
                rec.update(task=i, weight=1.0 / ((D - 1) * len(ts)))
                recs.append(rec)
    examples = {}
    for cls in QUESTION_CLASSES:
        rec = select_example(recs, cls, example)
        if rec is not None:
            task, scm = _task_and_scm(cfg, rec["task"], base_seed, num_variables)
            examples[cls] = (task, scm, rec["t"], rec["y"], rec)
    return recs, examples


def _bands(ax, task, scm, t, y, rec, legend):
    mu, S = scm.moments()
    xs = mu[t] + np.sqrt(S[t, t]) * np.linspace(-2.5, 2.5, 100)
    for (a, b, v), color, label in (
        (scm.conditional_line(t, y), COND_COLOR, "Y | T=x (90%)"),
        (scm.interventional_line(t, y), DO_COLOR, "Y | do(T=x) (90%)"),
    ):
        m, h = a + b * xs, Z90 * np.sqrt(v)
        ax.plot(xs, m, color=color, lw=2, label=label)
        ax.fill_between(xs, m - h, m + h, color=color, alpha=0.18)
    for target, color, size, name in (((), "#7F7F7F", 4, "obs"), ((t,), DO_COLOR, 7, "do(T)")):
        regs = [r for r, tg in enumerate(task.regime_targets) if tuple(tg) == target]
        rows = np.isin(task.ctx_regime, regs)
        ax.scatter(task.ctx_x[rows, t], task.ctx_x[rows, y], s=size, color=color,
                   alpha=0.5, label=f"context {name} rows")  # fmt: skip
    ax.set_xlim(xs[0], xs[-1])
    edges = ", ".join(f"X{i}→X{j}" for i, j in zip(*np.nonzero(task.graph)))
    ax.text(0.02, 0.02,
        f"D={len(task.graph)}, T=X{t}, Y=X{y}, task {rec['task']}\n"
        f"{edges or 'no edges'}\nW2/sd(Y) = {rec['w2_sd']:.2f}",
        transform=ax.transAxes, fontsize=6.5, va="bottom", wrap=True)  # fmt: skip
    ax.set_xlabel("treatment value x")
    if legend:
        ax.legend(fontsize=6.5, loc="upper left")


def plot_obs_vs_do(recs: list[dict], examples: dict, title: str):
    k_cls = len(QUESTION_CLASSES)
    fig, ax = plt.subplots(4, k_cls, figsize=(3.8 * k_cls, 13), squeeze=False)
    total = sum(r["weight"] for r in recs)
    wmax = max([r["w2_sd"] for r in recs] + [1e-3])
    emax = np.log10(1 + max([r["evidence_nats"] for r in recs] + [10]))
    for k, cls in enumerate(QUESTION_CLASSES):
        R = [r for r in recs if r["cls"] == cls]
        W = np.array([r["weight"] for r in R])
        share = W.sum() / total if total else 0.0
        differ = cls in DIFFER_CLASSES
        a = ax[0, k]
        if cls in examples:
            _bands(a, *examples[cls], legend=(k == 0))
        a.set_title(f"{cls}\n{share:.0%} of questions · Y|T {'≠' if differ else '='} Y|do(T)",
                    fontsize=9, color=DO_COLOR if differ else "black")  # fmt: skip
        if not R:
            for a in ax[1:, k]:
                a.axis("off")
            continue

        a = ax[1, k]
        a.hist([r["w2_sd"] for r in R], bins=30, range=(0, wmax), weights=W / W.sum(),
               color=DO_COLOR if differ else "#7F7F7F")  # fmt: skip
        if cls in examples:
            a.axvline(examples[cls][4]["w2_sd"], color="black", ls=":", lw=1)
        a.set_xlabel("W2(Y|T=x, Y|do(T=x)) / sd(Y)", fontsize=8)

        a = ax[2, k]
        n_rows = np.array([[r[f"n_{q}"] for q in CONTEXT_ROLES] for r in R], float)
        frac = (W[:, None] * n_rows / n_rows.sum(1, keepdims=True)).sum(0) / W.sum()
        left = 0.0
        for f, q, c in zip(frac, CONTEXT_ROLES, ROLE_COLORS):
            a.barh(0, f, left=left, color=c)
            if f > 0.07:
                a.text(left + f / 2, 0, f"{q}\n{f:.0%}", ha="center", va="center", fontsize=6.5)
            left += f
        mean_t = float((W * n_rows[:, CONTEXT_ROLES.index("do(T)")]).sum() / W.sum())
        a.set_xlim(0, 1)
        a.set_yticks([])
        a.set_xlabel(f"share of context rows (mean do(T) rows: {mean_t:.0f})", fontsize=8)

        a = ax[3, k]
        if differ:
            ev = np.array([r["evidence_nats"] for r in R])
            a.hist(np.log10(1 + ev), bins=30, range=(0, emax), weights=W / W.sum(), color=DO_COLOR)
            a.axvline(np.log10(1 + STRONG_NATS), color="black", ls="--", lw=1)
            none = W[np.array([r["n_do(T)"] for r in R]) == 0].sum() / W.sum()
            strong = W[ev >= STRONG_NATS].sum() / W.sum()
            a.text(0.97, 0.95, f"no do(T) rows: {none:.0%}\n≥{STRONG_NATS:g} nats: {strong:.0%}",
                   transform=a.transAxes, ha="right", va="top", fontsize=8)  # fmt: skip
            a.set_xlabel("log10(1 + evidence in do(T) rows, nats)", fontsize=8)
        else:
            a.text(0.5, 0.5, "do(T) rows match the\nobservational answer:\nno evidence needed",
                   ha="center", va="center", fontsize=8, transform=a.transAxes)  # fmt: skip
            a.axis("off")
    for r, label in enumerate(("example question", "distribution gap", "context composition",
                               "usefulness of do(T) rows")):  # fmt: skip
        ax[r, 0].set_ylabel(label, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ROLE_COLORS]
    fig.legend(handles, CONTEXT_ROLES, loc="lower center", ncol=len(CONTEXT_ROLES), fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def summary(recs: list[dict]) -> str:
    total = sum(r["weight"] for r in recs)
    d = [r for r in recs if r["cls"] in DIFFER_CLASSES]
    wd = sum(r["weight"] for r in d)
    strong = sum(r["weight"] for r in d if r["evidence_nats"] >= STRONG_NATS)
    useful = sum(r["weight"] * r["n_do(T)"] for r in d)
    n_int = sum(r["weight"] * sum(r[f"n_{q}"] for q in CONTEXT_ROLES[1:]) for r in recs)
    return (
        f"questions where Y|T != Y|do(T): {wd / total:.1%}; "
        f"of these, do(T) rows give >= {STRONG_NATS:g} nats: {strong / max(wd, 1e-12):.1%}; "
        f"interventional context rows that are do(T) for such a question: "
        f"{useful / max(n_int, 1e-12):.1%}"
    )


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--example", choices=EXAMPLE_MODES, default="median",
                        help="which question each row-1 panel shows")  # fmt: skip
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_tasks", type=int, default=2000)
    parser.add_argument("--num_variables", type=int, default=None,
                        help="fix D (default: D drawn from the config)")  # fmt: skip
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--out", default="obs_vs_do.pdf")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.function_generator != "linearnonidentifiable":
        parser.error("obs-vs-do is exact only for function_generator: linearnonidentifiable")
    recs, examples = collect_obs_vs_do(cfg, args.num_tasks, args.base_seed, 
                                       args.num_variables, args.example)
    for cls, ex in examples.items():
        print(f"example {cls}: task {ex[4]['task']}, T=X{ex[2]}, Y=X{ex[3]}, "
              f"W2/sd(Y)={ex[4]['w2_sd']:.3f}")
    text = summary(recs)
    print(text)

    d_text = f"D={args.num_variables}" if args.num_variables else f"D in {cfg.num_variables}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_obs_vs_do(recs, examples, f"{Path(args.config).name}, {d_text}, "
                                         f"{args.num_tasks} tasks\n{text}")  # fmt: skip
    fig.savefig(out)
    if args.png:
        fig.savefig(out.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(f"Saved {out}")
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"Saved {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
