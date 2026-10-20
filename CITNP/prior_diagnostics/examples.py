"""
Example tasks from the mixed-regime prior: one figure per task with the
causal graph (edge width = estimated standardised coefficient from the
observational context rows) and pairwise scatter plots of the context rows (coloured by
regime) and the query rows.

    python -m CITNP.prior_diagnostics examples \
        --config CITNP/datasets/configs/mixed_linear_gaussian.yaml \
        --num_tasks 12 --out prior_examples.pdf [--num_variables 3] [--png]

Complements summary.py, which plots aggregate statistics over many
tasks. Both use SOURCE_ADHOC keys, so task i here is task i there for the
same --base_seed.

Device: GP and NN mechanisms run on CUDA when it is visible, and CPU and CUDA
random streams differ. To see exactly the tasks a CPU chunk writer would make
for those families, run with CUDA_VISIBLE_DEVICES="". Linear families are
CPU-only and unaffected.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_ADHOC,
    MixedRegimeConfig,
    Task,
    generate_task,
    make_key,
)
from CITNP.prior_diagnostics.stats import edge_strengths

__all__ = [
    "plot_graph",
    "plot_task",
    "regime_label",
    "sample_example_tasks",
    "save_examples",
]

OBS_COLOR = "#8C8C8C"
QRY_COLOR = "black"
CTX_TARGET_COLOR = "#DD8452"
QRY_TARGET_COLOR = "#C44E52"
POS_EDGE_COLOR = "#2B6CB0"
NEG_EDGE_COLOR = "#C53030"
MIN_OBS_ROWS_FOR_WEIGHTS = 20


def regime_label(targets: tuple[int, ...]) -> str:
    """'obs' for the observational regime, else 'do(X_i, ...)'."""
    if len(targets) == 0:
        return "obs"
    return "do(" + ", ".join(f"X{t}" for t in targets) + ")"


def sample_example_tasks(
    cfg: MixedRegimeConfig,
    num_tasks: int,
    base_seed: int = 0,
    num_variables: int | None = None,
) -> list[Task]:
    """Draw `num_tasks` tasks with keys make_key(SOURCE_ADHOC, 0, base_seed, i)."""
    return [
        generate_task(
            cfg, make_key(SOURCE_ADHOC, 0, base_seed, i), num_variables=num_variables
        )
        for i in range(num_tasks)
    ]


def plot_graph(
    task: Task, ax: plt.Axes, edge_weights: np.ndarray | None = None
) -> None:
    """Draw the task's DAG on a circle. Fill: intervened in some context
    regime; red ring: intervened in some query row. With `edge_weights`
    ([D, D], e.g. stats.edge_strengths), edge width is proportional to
    |weight|, blue for positive and red for negative, with the value shown."""
    D = task.graph.shape[0]
    angle = np.pi / 2 - 2 * np.pi * np.arange(D) / D
    pos = np.c_[np.cos(angle), np.sin(angle)]
    ctx_targets = {t for r in task.regime_targets for t in r}
    qry_targets = set(np.flatnonzero(task.qry_int.any(axis=0)).tolist())
    wmax = None
    if edge_weights is not None and np.abs(edge_weights).max() > 0:
        wmax = float(np.abs(edge_weights).max())

    for i, j in zip(*np.nonzero(task.graph)):
        if wmax is None:
            lw, color = 1.2, "#333333"
        else:
            w = float(edge_weights[i, j])
            lw = 0.6 + 4.4 * abs(w) / wmax
            color = POS_EDGE_COLOR if w >= 0 else NEG_EDGE_COLOR
            mid = 0.55 * pos[i] + 0.45 * pos[j]
            ax.text(*mid, f"{w:+.2f}", fontsize=7, ha="center", va="center",
                    zorder=5, bbox=dict(fc="white", ec="none", pad=0.5))  # fmt: skip
        ax.annotate(
            "",
            xy=pos[j],
            xytext=pos[i],
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=16,
                            shrinkB=16, mutation_scale=10 + 2 * lw),
        )  # fmt: skip
    for j in range(D):
        in_qry = j in qry_targets
        ax.scatter(
            *pos[j],
            s=700,
            c=CTX_TARGET_COLOR if j in ctx_targets else "white",
            edgecolors=QRY_TARGET_COLOR if in_qry else "#333333",
            linewidths=2.5 if in_qry else 1.2,
            zorder=3,
        )
        ax.text(*pos[j], f"X{j}", ha="center", va="center", zorder=4)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    title = "graph\n(filled: intervened in context, red ring: query treatment)"
    if wmax is not None:
        title += "\nedge width ∝ |standardised coef| (blue +, red −)"
    ax.set_title(title, fontsize=8)


def _pair_title(graph: np.ndarray, i: int, j: int) -> str:
    if graph[i, j]:
        return f"X{i} → X{j}"
    if graph[j, i]:
        return f"X{j} → X{i}"
    return f"X{i}, X{j} (no edge)"


def plot_task(task: Task, max_pairs: int = 6, task_id: int | None = None) -> Figure:
    """One figure: the graph, then up to `max_pairs` variable pairs (edges
    first). Context rows are dots coloured by regime; query rows (true values,
    do-cells hold the do-value) are black crosses."""
    D = task.graph.shape[0]
    edges = {(min(i, j), max(i, j)) for i, j in zip(*np.nonzero(task.graph))}
    pairs = sorted(combinations(range(D), 2), key=lambda p: p not in edges)
    pairs = pairs[:max_pairs]

    n_panels = 1 + len(pairs)
    ncols = min(4, n_panels)
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.9 * nrows), squeeze=False
    )
    ax = axes.ravel()
    obs_r = [r for r, t in enumerate(task.regime_targets) if len(t) == 0]
    x_obs = task.ctx_x[task.ctx_regime == obs_r[0]] if obs_r else task.ctx_x[:0]
    weights = (
        edge_strengths(x_obs, task.graph)
        if len(x_obs) >= MIN_OBS_ROWS_FOR_WEIGHTS
        else None
    )
    plot_graph(task, ax[0], edge_weights=weights)

    cmap = plt.get_cmap("tab10")
    for k, (i, j) in enumerate(pairs, start=1):
        a = ax[k]
        for r, targets in enumerate(task.regime_targets):
            rows = task.ctx_regime == r
            if not rows.any():
                continue
            a.scatter(
                task.ctx_x[rows, i],
                task.ctx_x[rows, j],
                s=6,
                alpha=0.5,
                color=OBS_COLOR if len(targets) == 0 else cmap(r % 10),
                label=f"{regime_label(targets)} (n={int(rows.sum())})",
            )
        a.scatter(
            task.qry_x[:, i],
            task.qry_x[:, j],
            s=14,
            marker="x",
            linewidths=0.8,
            color=QRY_COLOR,
            label=f"query (n={len(task.qry_x)})",
        )
        a.set_xlabel(f"X{i}")
        a.set_ylabel(f"X{j}")
        a.set_title(_pair_title(task.graph, i, j), fontsize=9)
    if len(pairs) > 0:
        ax[1].legend(fontsize=7, markerscale=2, loc="best")
    for a in ax[n_panels:]:
        a.axis("off")

    m = task.meta
    head = f"task {task_id} | " if task_id is not None else ""
    fig.suptitle(
        f"{head}D={D}, K={m.get('K')}, has_obs={m.get('has_obs')}, "
        f"N_ctx={len(task.ctx_x)}, graph={m.get('graph_type')}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def save_examples(
    tasks: list[Task], out: str | Path, max_pairs: int = 6, png: bool = False
) -> list[Path]:
    """Write a multi-page PDF (one page per task) and optionally one PNG per
    task next to it (<stem>_<i>.png). Returns all written paths."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = [out]
    with PdfPages(out) as pdf:
        for idx, task in enumerate(tasks):
            fig = plot_task(task, max_pairs=max_pairs, task_id=idx)
            pdf.savefig(fig)
            if png:
                p = out.with_name(f"{out.stem}_{idx:03d}.png")
                fig.savefig(p, dpi=120)
                written.append(p)
            plt.close(fig)
    return written


def describe(task: Task, idx: int) -> str:
    """One-line text summary of a task, printed by the CLI."""
    edges = ", ".join(f"X{i}→X{j}" for i, j in zip(*np.nonzero(task.graph)))
    regimes = ", ".join(
        f"{regime_label(t)}:{int((task.ctx_regime == r).sum())}"
        for r, t in enumerate(task.regime_targets)
    )
    return f"task {idx}: edges [{edges or 'none'}] | regimes [{regimes}]"


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_tasks", type=int, default=12)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--num_variables", type=int, default=None,
                        help="fix D (must be allowed by the config's graph_degrees)")  # fmt: skip
    parser.add_argument("--max_pairs", type=int, default=6)
    parser.add_argument("--out", default="prior_examples.pdf")
    parser.add_argument("--png", action="store_true",
                        help="also write one PNG per task next to --out")  # fmt: skip
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tasks = sample_example_tasks(
        cfg, args.num_tasks, args.base_seed, args.num_variables
    )
    for idx, task in enumerate(tasks):
        print(describe(task, idx))
    for p in save_examples(tasks, args.out, args.max_pairs, args.png):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
