"""
Population-level diagnostics of the prior DGP: where do the training
(treatment, outcome) pairs sit in the DAG, and how large are the induced
causal effects and confounding gaps?

    python -m CITNP.prior_diagnostics stats \
        --config CITNP/datasets/configs/mixed_linear_gaussian.yaml \
        --num_tasks 2000 --n_probe_tasks 300 --out prior_stats.pdf

Writes one figure (PDF, plus PNG with --png) and two CSV files next to it:
<stem>_pairs.csv (one row per (task, treatment, outcome) pair, weighted so
each task sums to 1) and <stem>_effects.csv (one row per probed pair).

Panels:
  1. Relation of query outcome to query treatment by D (weighted fractions):
     direct child, downstream at distance 2 or 3+, upstream, confounded only,
     independent. Upstream / confounded / independent pairs have no causal
     effect, so a large share means most training queries are "do(X_t) does
     nothing to X_y".
  2. The same for context interventions vs every other node.
  3. |E[X_y | do(X_t=x)] - E[X_y]| in observational sd of X_y, by relation:
     how much signal an intervention carries.
  4. |E[X_y | do(X_t=x)] - E[X_y | X_t~=x]|: how different intervening is
     from conditioning (confounding), by relation.
Panels 3-4 use probes (see obs_vs_int.py) on the first n_probe_tasks tasks,
treatment drawn from the task's own query treatments, x at the --quantile
observational quantile of X_t.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import SOURCE_ADHOC, generate_task, make_key
from CITNP.prior_diagnostics.obs_vs_int import probe_task
from CITNP.prior_diagnostics.stats import (
    RELATIONS,
    context_pair_records,
    effect_gaps,
    path_lengths,
    query_pair_records,
    query_treatments,
    relation,
)

__all__ = ["collect_effects", "collect_pairs", "plot_stats"]

REL_COLORS = ["#08519C", "#3182BD", "#9ECAE1", "#FDAE6B", "#E6550D", "#BDBDBD"]


def collect_pairs(cfg, num_tasks: int, base_seed: int = 0) -> list[dict]:
    recs = []
    for i in trange(num_tasks, desc="pairs"):
        task = generate_task(cfg, make_key(SOURCE_ADHOC, 0, base_seed, i))
        for r in query_pair_records(task) + context_pair_records(task):
            recs.append({"task": i, **r})
    return recs


def collect_effects(
    cfg, n_tasks: int, base_seed: int = 0, quantile: float = 0.84,
    n_rows: int = 500, knn: int = 50,
) -> list[dict]:  # fmt: skip
    rng = np.random.default_rng(base_seed)
    recs = []
    for i in trange(n_tasks, desc="probes"):
        base = generate_task(cfg, make_key(SOURCE_ADHOC, 0, base_seed, i))
        ts = query_treatments(base)
        if not ts:
            continue
        t = int(rng.choice(ts))
        probe, _ = probe_task(cfg, i, base_seed, treatment=t,
                              do_quantiles=(quantile,), n_obs=n_rows, n_per_do=n_rows)  # fmt: skip
        dist = path_lengths(probe.graph)
        x_obs = probe.ctx_x[probe.ctx_regime == 0]
        x_int = probe.ctx_x[probe.ctx_regime == 1]
        x = float(x_int[0, t])
        for y in range(len(dist)):
            if y != t:
                g = effect_gaps(x_obs, x_int, t, y, x, knn)
                recs.append({
                    "task": i, "D": len(dist), "t": t, "y": y, "do_value": x,
                    "relation": relation(dist, t, y),
                    "int_shift": g["int_shift"], "int_vs_cond": g["int_vs_cond"],
                })  # fmt: skip
    return recs


def _fractions(recs, source):
    """{D: {relation: weighted fraction}}."""
    acc = defaultdict(lambda: defaultdict(float))
    tasks = defaultdict(set)
    for r in recs:
        if r["source"] == source:
            acc[r["D"]][r["relation"]] += r["weight"]
            tasks[r["D"]].add(r["task"])
    return {D: {k: v / len(tasks[D]) for k, v in acc[D].items()} for D in sorted(acc)}


def _stacked(ax, frac, title):
    Ds = list(frac)
    bottom = np.zeros(len(Ds))
    for rel, c in zip(RELATIONS, REL_COLORS):
        vals = np.array([frac[D].get(rel, 0.0) for D in Ds])
        ax.bar([str(D) for D in Ds], vals, bottom=bottom, color=c, label=rel)
        bottom += vals
    ax.set_xlabel("D (num variables)")
    ax.set_ylabel("fraction of pairs")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9)


def _boxes(ax, effects, key, title):
    data = [[abs(e[key]) for e in effects if e["relation"] == rel] for rel in RELATIONS]
    keep = [i for i, d in enumerate(data) if d]
    if keep:
        bp = ax.boxplot([data[i] for i in keep], showfliers=False, patch_artist=True)
        for patch, i in zip(bp["boxes"], keep):
            patch.set_facecolor(REL_COLORS[i])
        ax.set_xticks(range(1, len(keep) + 1),
                      [f"{RELATIONS[i]}\n(n={len(data[i])})" for i in keep],
                      fontsize=7, rotation=30)  # fmt: skip
    ax.set_ylabel("|gap| (obs sd of X_y)")
    ax.set_title(title, fontsize=9)


def plot_stats(pairs, effects, title: str):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    _stacked(ax[0, 0], _fractions(pairs, "query"), "query treatment → outcome (training pairs)")
    _stacked(ax[0, 1], _fractions(pairs, "context"), "context intervention → other nodes")
    ax[0, 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    _boxes(ax[1, 0], effects, "int_shift", "causal effect: E[Y|do(x)] − E[Y]")
    _boxes(ax[1, 1], effects, "int_vs_cond", "do vs conditional: E[Y|do(x)] − E[Y|X≈x]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _write_csv(path: Path, recs: list[dict]) -> None:
    if recs:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
        print(f"Saved {path}")


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_tasks", type=int, default=2000)
    parser.add_argument("--n_probe_tasks", type=int, default=300)
    parser.add_argument("--n_rows", type=int, default=500,
                        help="rows per regime in each probe")  # fmt: skip
    parser.add_argument("--quantile", type=float, default=0.84,
                        help="do-value at this observational quantile of X_t (0.84 ≈ +1 sd)")  # fmt: skip
    parser.add_argument("--knn", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--out", default="prior_stats.pdf")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    pairs = collect_pairs(cfg, args.num_tasks, args.base_seed)
    effects = collect_effects(cfg, min(args.n_probe_tasks, args.num_tasks),
                              args.base_seed, args.quantile, args.n_rows, args.knn)  # fmt: skip
    for source in ("query", "context"):
        print(f"[{source}] weighted relation fractions by D:")
        for D, f in _fractions(pairs, source).items():
            print(f"  D={D}: " + ", ".join(f"{k}={f.get(k, 0):.2f}" for k in RELATIONS))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_stats(pairs, effects, f"{cfg.function_generator}: {args.num_tasks} tasks, "
                                     f"{args.n_probe_tasks} probes")  # fmt: skip
    fig.savefig(out)
    print(f"Saved {out}")
    if args.png:
        fig.savefig(out.with_suffix(".png"), dpi=120)
        print(f"Saved {out.with_suffix('.png')}")
    plt.close(fig)
    _write_csv(out.with_name(f"{out.stem}_pairs.csv"), pairs)
    _write_csv(out.with_name(f"{out.stem}_effects.csv"), effects)


if __name__ == "__main__":
    main()
