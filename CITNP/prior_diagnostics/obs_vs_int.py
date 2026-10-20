"""
Interventional vs observational-conditional distributions for ONE instance
(one SCM) of the mixed-regime prior DGP.

For a treatment X_t and a few do-values x (by default the 10/50/90%
quantiles of the observational X_t), each panel shows, for one outcome X_y:
  - p(X_y)                  observational marginal (thin grey)
  - p(X_y | X_t ~= x)       observational conditional, k nearest rows (dashed)
  - p(X_y | do(X_t = x))    interventional (solid)
plus the standardised mean gaps of stats.effect_gaps. The graph panel
shows edge widths from stats.edge_strengths on the observational rows.

    python -m CITNP.prior_diagnostics obs-vs-int \
        --config CITNP/experiments/configs/oracle_d3.yaml \
        --num_tasks 4 --out obs_vs_int.pdf [--png] \
        [--treatment 0] [--do_quantiles 0.1,0.5,0.9 | --do_values=-2,0,2]

How an instance is built: task i is drawn as usual (key
make_key(SOURCE_ADHOC, 0, base_seed, i), the same task as in
examples.py / summary.py). Its graph is kept and the same key
generates a probe on that graph: one large observational regime plus one
large do(X_t = x) regime per x, in ONE generation call, so every curve comes
from the same SCM. Mechanisms are a fresh draw from the prior on that graph,
not those of task i (they are not stored). Labels match task i.

With --do_quantiles, a pilot probe with the same key and regime sizes gives
the observational quantiles; fixed do-values do not change the random stream,
so the probe has the same mechanisms (and, for non-GP families, the same
observational rows).

Requires Design.regime_do_values and generate_task(permute=...). GP families:
n_ref + n_obs + n_per_do * len(x) must stay within max_total_rows_gp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_ADHOC,
    Design,
    MixedRegimeConfig,
    Task,
    descendants,
    generate_task,
    make_key,
)
from CITNP.prior_diagnostics.examples import OBS_COLOR, plot_graph
from CITNP.prior_diagnostics.stats import (
    conditional_rows,
    edge_strengths,
    effect_gaps,
    path_lengths,
    relation,
)

__all__ = [
    "default_treatment",
    "make_probe_design",
    "plot_obs_vs_int",
    "probe_task",
]

COND_COLOR = "#4C72B0"
INT_COLOR = "#C44E52"


def default_treatment(graph: np.ndarray) -> int:
    """Node with the most descendants (lowest index on ties)."""
    return int(np.argmax([len(descendants(graph, j)) for j in range(len(graph))]))


def make_probe_design(
    D: int,
    treatment: int,
    do_values,
    n_obs: int = 2000,
    n_per_do: int = 2000,
) -> Design:
    """Context = [obs regime, do(X_t = x_1), do(X_t = x_2), ...]."""
    do_values = [float(x) for x in do_values]
    return Design(
        num_variables=D,
        regime_targets=[()] + [(treatment,)] * len(do_values),
        regime_sizes=[n_obs] + [n_per_do] * len(do_values),
        do_dist={"kind": "normal", "scale": 1.0},  # overwritten by do_values
        qry_targets=[()],
        regime_do_values=[None, *do_values],
    )


def probe_task(
    cfg: MixedRegimeConfig,
    index: int,
    base_seed: int = 0,
    num_variables: int | None = None,
    treatment: int | None = None,
    do_values=None,
    do_quantiles=(0.1, 0.5, 0.9),
    n_obs: int = 2000,
    n_per_do: int = 2000,
) -> tuple[Task, int]:
    """Probe on the graph of prior task `index` (labels as in that task).
    Fixed `do_values` win; otherwise do-values are observational quantiles
    of X_t. Returns (probe, treatment)."""
    key = make_key(SOURCE_ADHOC, 0, base_seed, index)
    base = generate_task(cfg, key, num_variables=num_variables)
    D = base.graph.shape[0]
    t = default_treatment(base.graph) if treatment is None else int(treatment)
    if not 0 <= t < D:
        raise ValueError(f"treatment {t} out of range for D={D}")

    def make(values):
        design = make_probe_design(D, t, values, n_obs, n_per_do)
        return generate_task(cfg, key, graph=base.graph, design=design, permute=False)

    if do_values is None:
        pilot = make([0.0] * len(do_quantiles))
        do_values = np.quantile(pilot.ctx_x[pilot.ctx_regime == 0, t], do_quantiles)
    return make(do_values), t


def _density(ax: plt.Axes, y: np.ndarray, grid: np.ndarray, **kw) -> None:
    if len(y) < 2 or np.std(y) < 1e-9:  # point mass, e.g. deterministic node
        ax.axvline(float(np.mean(y)), **kw)
    else:
        ax.plot(grid, gaussian_kde(y)(grid), **kw)


def plot_obs_vs_int(
    probe: Task,
    treatment: int,
    task_id: int | None = None,
    knn: int | None = None,
    max_outcomes: int = 5,
) -> Figure:
    """Weighted graph on the left; a grid with one row per do-value and one
    column per outcome (closest downstream outcomes first)."""
    t = treatment
    D = probe.graph.shape[0]
    dist = path_lengths(probe.graph)
    outcomes = sorted((y for y in range(D) if y != t), key=lambda y: (dist[t, y], y))
    outcomes = outcomes[:max_outcomes]
    x_obs = probe.ctx_x[probe.ctx_regime == 0]
    k = knn or max(50, len(x_obs) // 20)
    regimes = range(1, len(probe.regime_targets))
    do_vals = [float(probe.ctx_x[probe.ctx_regime == r, t][0]) for r in regimes]

    nrows, ncols = len(do_vals), len(outcomes)
    fig = plt.figure(figsize=(3.4 * ncols + 4.5, 2.9 * nrows + 0.8))
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1.4] + [1] * ncols)
    plot_graph(probe, fig.add_subplot(gs[:, 0]), edge_strengths(x_obs, probe.graph))

    for row, (r, x) in enumerate(zip(regimes, do_vals)):
        x_int = probe.ctx_x[probe.ctx_regime == r]
        idx = conditional_rows(x_obs[:, t], x, k)
        for col, y in enumerate(outcomes):
            a = fig.add_subplot(gs[row, col + 1])
            both = np.r_[x_obs[:, y], x_int[:, y]]
            lo, hi = np.percentile(both, [0.5, 99.5])
            pad = 0.1 * (hi - lo + 1e-9)
            grid = np.linspace(lo - pad, hi + pad, 300)
            _density(a, x_obs[:, y], grid, color=OBS_COLOR, lw=1, alpha=0.7,
                     label=f"p(X{y})")  # fmt: skip
            _density(a, x_obs[idx, y], grid, color=COND_COLOR, lw=2, ls="--",
                     label=f"p(X{y} | X{t}≈x), k={len(idx)}")  # fmt: skip
            _density(a, x_int[:, y], grid, color=INT_COLOR, lw=2,
                     label=f"p(X{y} | do(X{t}=x))")  # fmt: skip
            g = effect_gaps(x_obs, x_int, t, y, x, k)
            a.text(0.02, 0.97,
                   f"do−marg {g['int_shift']:+.2f} sd\ndo−cond {g['int_vs_cond']:+.2f} sd",
                   transform=a.transAxes, va="top", fontsize=7)  # fmt: skip
            a.set_title(f"X{y} | do(X{t}={x:.2f}) [{relation(dist, t, y)}]", fontsize=8)
            a.set_yticks([])
            if row == 0 and col == 0:
                a.legend(fontsize=6, loc="upper right")

    m = probe.meta
    head = f"task {task_id} | " if task_id is not None else ""
    fig.suptitle(
        f"{head}{m.get('family')}, D={D}, treatment X{t}, "
        f"{len(x_obs)} rows per distribution; gaps in observational sd of X_y",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def _floats(s: str | None):
    return None if s is None else tuple(float(v) for v in s.split(","))


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_tasks", type=int, default=4,
                        help="tasks 0..num_tasks-1, one page each")  # fmt: skip
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--num_variables", type=int, default=None)
    parser.add_argument("--treatment", type=int, default=None,
                        help="default: node with most descendants")  # fmt: skip
    parser.add_argument("--do_values", default=None,
                        help="fixed do-values, e.g. --do_values=-2,0,2 (wins over quantiles)")  # fmt: skip
    parser.add_argument("--do_quantiles", default="0.1,0.5,0.9",
                        help="do-values at these observational quantiles of X_t")  # fmt: skip
    parser.add_argument("--n_obs", type=int, default=2000)
    parser.add_argument("--n_per_do", type=int, default=2000)
    parser.add_argument("--knn", type=int, default=None,
                        help="rows for the conditional; default max(50, n_obs/20)")  # fmt: skip
    parser.add_argument("--max_outcomes", type=int, default=5)
    parser.add_argument("--out", default="obs_vs_int.pdf")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for i in range(args.num_tasks):
            probe, t = probe_task(
                cfg, i, args.base_seed, args.num_variables, args.treatment,
                _floats(args.do_values), _floats(args.do_quantiles),
                args.n_obs, args.n_per_do,
            )  # fmt: skip
            fig = plot_obs_vs_int(probe, t, i, args.knn, args.max_outcomes)
            pdf.savefig(fig)
            if args.png:
                p = out.with_name(f"{out.stem}_{i:03d}.png")
                fig.savefig(p, dpi=120)
                print(f"Saved {p}")
            plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
