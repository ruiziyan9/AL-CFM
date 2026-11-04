"""Command-line entry point for model evaluation (argument parsing only).

    python -m CITNP.evaluation kshot --ckpt ... --model_config ... --data_config ... \
           [--k_list 0,1,3,5,10,20,50] [--n_seeds 100] [--random_weights] --out kshot.csv
    python -m CITNP.evaluation kshot-prior --ckpt ... --model_config ... \
           --data_config ... [--num_variables 3,4] [--n_tasks 500] \
           [--task_offset 0] [--k_list 0,1,3,5,10] --out kshot_prior.csv
    python -m CITNP.evaluation scm-suite --ckpt ... --model_config ... \
           --data_config ... --out suite.csv
    python -m CITNP.evaluation bank --data_config ... --out bank.pkl
    python -m CITNP.evaluation calibrate --data_config ... --n_tasks 2000 \
           [--n_mc 1000] [--compare_mc] --out calib.csv   # oracle SBC: PIT + MEC posterior
    python -m CITNP.evaluation plot-kshot kshot.csv     # all k-shot figures, no GPU
    python -m CITNP.evaluation plot-kshot-prior part0.csv [part1.csv ...] \
           [--prefix P]            # merges array-job parts (same settings)
    python -m CITNP.evaluation plot-monitoring RUN_DIR [RUN_DIR ...] --out monitoring.png
    python -m CITNP.evaluation probe-flag --ckpt ... --model_config ... \
           --bank bank.pkl --out probe.csv               # + probe.png
    python -m CITNP.evaluation embedding-geometry --ckpt ... \
           --model_config ... --out geometry.json      # extra_node only
    python -m CITNP.evaluation probe-intervention --ckpt ... \
           --model_config ... --data_config ... [--n_tasks 200] \
           --out intervention_probes.csv               # + _meta.json
    python -m CITNP.evaluation context-edits --ckpt ... --model_config ... \
           --data_config ... [--n_tasks 200] [--max_edit_steps 16] \
           --out context_edits.csv     # + _summary.csv, _meta.json

`kshot` evaluates ONE hand-built SCM, where the model can beat the oracle.
ONLY `kshot-prior` gives a true oracle lower bound: every task is drawn from
the training prior, so E[KL(truth||model)] - E[KL(truth||oracle)] =
E[KL(oracle||model)] >= 0. Its prior comes from the data config (graph,
variable permutation, weights, noise, do-value distribution, D, default
n_obs, oracle priors); the CLI only sets the protocol (k_list, n_tasks,
task_offset, base_seed, n_y, n_mc, grid_eps) and how many graphs estimate
the oracle's graph prior (graph_prior_samples).

See CITNP.evaluation.experiments for what `kshot`, `kshot-prior`, `scm-suite`
and `probe-flag` compute, CITNP.evaluation.embedding_geometry for
`embedding-geometry`, CITNP.evaluation.intervention_diagnostics for
`probe-intervention` and `context-edits`,
CITNP.evaluation.tasks.build_bank for the bank and CITNP.evaluation.plots for
the figures.
"""

import argparse
import sys


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_args(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument("--ckpt", required=required)
    p.add_argument("--model_config", required=required)
    p.add_argument("--data_config", required=required)


# --------------------------------------------------------------------------- #
# Parsers (defaults as in the original scripts)
# --------------------------------------------------------------------------- #
def _kshot_common_args(
    p: argparse.ArgumentParser, *, n_mc: int, out: str, k_list_help: str
) -> None:
    """Arguments shared by `kshot` and `kshot-prior`."""
    _model_args(p, required=False)
    p.add_argument("--k_list", default=None, help=k_list_help)
    p.add_argument("--n_t", type=int, default=13, help="query grid points")
    p.add_argument(
        "--n_y", type=int, default=64, help="MC samples per grid point for KLs"
    )
    p.add_argument(
        "--n_mc", type=int, default=n_mc, help="oracle parameter draws per DAG"
    )
    p.add_argument("--min_graph_prob", type=float, default=1e-4)
    p.add_argument("--out", default=out)
    p.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )


def _kshot_parser(p: argparse.ArgumentParser) -> None:
    _kshot_common_args(
        p,
        n_mc=200,
        out="kshot.csv",
        k_list_help="comma-separated numbers of experiments, e.g. 0,1,3,5,10; "
        "overrides --k_max (default: 0..k_max)",
    )
    p.add_argument(
        "--plot_only", default=None, help="CSV from a previous run: only plot"
    )
    p.add_argument("--n_seeds", type=int, default=100)
    p.add_argument("--n_obs", type=int, default=128)
    p.add_argument("--k_max", type=int, default=10)
    p.add_argument("--random_weights", action="store_true",
                   help="fresh confounder-SCM weights per seed (same graph)")  # fmt: skip
    p.add_argument("--do_scale", type=float, default=1.5,
                   help="sd of do-values in the context's interventional rows")  # fmt: skip
    p.add_argument(
        "--t_max", type=float, default=3.0, help="query grid is [-t_max, t_max]"
    )


def _kshot_prior_parser(p: argparse.ArgumentParser) -> None:
    _kshot_common_args(
        p,
        n_mc=2000,
        out="kshot_prior.csv",
        k_list_help="comma-separated numbers of interventional rows "
        "(default 0,1,3,5,10; 0 is always added)",
    )
    p.add_argument("--num_variables", default=None,
                   help="comma-separated D values, a subset of the data "
                        "config's num_variables (default: all)")  # fmt: skip
    p.add_argument("--n_tasks", type=int, default=500)
    p.add_argument("--task_offset", type=int, default=0,
                   help="first task index (split work across array jobs)")  # fmt: skip
    p.add_argument("--base_seed", type=int, default=9101)
    p.add_argument("--n_obs", type=int, default=None,
                   help="observational rows (default: from the data config)")  # fmt: skip
    p.add_argument("--grid_eps", type=float, default=0.02,
                   help="do-value grid: quantiles [eps, 1 - eps] of each "
                        "task's do_dist")  # fmt: skip
    p.add_argument("--flag_ablation", action="store_true",
                   help="also score the model with intervention flags zeroed")  # fmt: skip
    p.add_argument("--max_oracle_nodes", type=int, default=4,
                   help="largest D for exact DAG enumeration")  # fmt: skip
    p.add_argument("--graph_prior_samples", type=int, default=None,
                   help="graphs sampled for the oracle's graph prior "
                        "(default: 20000 for D = 3, 500000 "
                        "for D >= 4)")  # fmt: skip
    p.add_argument("--oracle_budget", choices=["per_dag", "total"],
                   default="total",
                   help="oracle MC draws: n_mc in total (default) or n_mc "
                        "per retained DAG")  # fmt: skip


def _plot_kshot_prior_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("csvs", nargs="+",
                   help="<out>.csv parts written by kshot-prior "
                        "(<out>_meta.json must sit next to each)")  # fmt: skip
    p.add_argument("--prefix", default=None,
                   help="figure prefix (default: first CSV without _part<N>)")  # fmt: skip


def _suite_parser(p: argparse.ArgumentParser) -> None:
    _model_args(p, required=True)
    p.add_argument("--n_seeds", type=int, default=30)
    p.add_argument("--n_query", type=int, default=256)
    p.add_argument("--do_scale", type=float, default=1.5)
    p.add_argument("--n_mc", type=int, default=1000)
    p.add_argument("--out", default="results_d3_oracle.csv")
    p.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )


def _bank_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per_bucket", type=int, default=150)
    p.add_argument("--n_mc", type=int, default=2000)
    p.add_argument("--seed", type=int, default=4242)


def _calibrate_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_config", required=True)
    p.add_argument("--n_tasks", type=int, default=2000)
    p.add_argument("--n_mc", type=int, default=1000,
                   help="oracle parameter draws per DAG")  # fmt: skip
    p.add_argument("--compare_mc", action="store_true",
                   help="also run with 2x draws and report mean |delta NLL|")  # fmt: skip
    p.add_argument("--seed", type=int, default=7001)
    p.add_argument("--out", default="calib.csv")


def _plot_kshot_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "csv", help="<prefix>.csv written by kshot (reads <prefix>_by_t.csv too)"
    )


def _plot_monitoring_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default="monitoring.png")


def _probe_flag_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt")
    p.add_argument("--model_config")
    p.add_argument("--bank", help="pickle from `python -m CITNP.evaluation bank`")
    p.add_argument(
        "--plot_only", default=None, help="CSV from a previous run: only plot"
    )
    p.add_argument("--max_contexts", type=int, default=400)
    p.add_argument("--cells_per_context", type=int, default=64)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_normalise", action="store_true",
                   help="skip per-task standardisation (as in training)")  # fmt: skip
    p.add_argument("--out", default="probe.csv")
    p.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )


def _embedding_geometry_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_config", required=True,
                   help="model YAML of an extra_node model")  # fmt: skip
    p.add_argument("--num_variables", type=int, default=3,
                   help="number of variables D the model is built for")  # fmt: skip
    p.add_argument("--out", default="metadata_embedding_geometry.json")
    p.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )


def _diagnostic_data_args(p: argparse.ArgumentParser, *, base_seed: int) -> None:
    """Model, data and task-stream arguments of the intervention-encoding
    diagnostics that draw tasks from a prior."""
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--data_config", required=True,
                   help="prior config, or a pre-simulated dataset config "
                        "(its prior_config is used; chunk files are not "
                        "read)")  # fmt: skip
    p.add_argument("--num_variables", type=int, default=None,
                   help="D (default: the smallest D of the prior)")  # fmt: skip
    p.add_argument("--n_tasks", type=int, default=200)
    p.add_argument("--task_offset", type=int, default=0)
    p.add_argument("--base_seed", type=int, default=base_seed,
                   help="task stream seed (prior_task key)")  # fmt: skip
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_normalise", action="store_true",
                   help="skip per-task standardisation (as in training)")  # fmt: skip
    p.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )


def _probe_intervention_parser(p: argparse.ArgumentParser) -> None:
    _diagnostic_data_args(p, base_seed=9201)
    p.add_argument("--rows_per_task", type=int, default=32)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--out", default="intervention_probes.csv")


def _context_edits_parser(p: argparse.ArgumentParser) -> None:
    _diagnostic_data_args(p, base_seed=9301)
    p.add_argument("--n_mc", type=int, default=2000,
                   help="oracle MC draws per query (budget total)")  # fmt: skip
    p.add_argument("--min_graph_prob", type=float, default=1e-4)
    p.add_argument("--max_query_points", type=int, default=32,
                   help="query rows used per treatment")  # fmt: skip
    p.add_argument("--graph_prior_samples", type=int, default=None,
                   help="graphs sampled for the oracle's graph prior "
                        "(default: 20000 for D = 3, 500000 "
                        "for D >= 4)")  # fmt: skip
    p.add_argument("--max_oracle_nodes", type=int, default=4,
                   help="largest D for exact DAG enumeration")  # fmt: skip
    p.add_argument("--n_y", type=int, default=64,
                   help="MC draws per do-value for the KLs")  # fmt: skip
    p.add_argument("--max_edit_steps", type=int, default=None,
                   help="at most this many numbers of edited rows per task "
                        "(geometric; default: every n from 1 to all)")  # fmt: skip
    p.add_argument("--keep_original_normalisation", action="store_true",
                   help="give every edited context the unedited context's "
                        "standardisation and scale features, so only the "
                        "mask differs (primary readout for the encoding "
                        "question; default: from the edited mask, the "
                        "model's normal input path)")  # fmt: skip
    p.add_argument("--out", default="context_edits.csv")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def _plot_kshot(csv_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from CITNP.evaluation.plots import plot_kshot_all

    plot_kshot_all(csv_path)


def _run_kshot(a: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import pandas as pd

    k_list = (sorted({int(k) for k in a.k_list.split(",")}) if a.k_list
              else list(range(a.k_max + 1)))  # fmt: skip
    if a.plot_only:
        _plot_kshot(a.plot_only)
        return
    if not (a.ckpt and a.model_config and a.data_config):
        parser.error("--ckpt, --model_config and --data_config are required")

    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.evaluation.experiments import run_kshot
    from CITNP.evaluation.model_api import load_model
    from CITNP.evaluation.tasks import MAIN_QUERY

    device = a.device or _default_device()
    data_cfg = load_config(a.data_config)
    model = load_model(a.ckpt, a.model_config, device)
    df, df_t = run_kshot(
        model, data_cfg, device=device, k_list=k_list, n_seeds=a.n_seeds,
        n_obs=a.n_obs, random_weights=a.random_weights, do_scale=a.do_scale,
        t_max=a.t_max, n_t=a.n_t, n_y=a.n_y, n_mc=a.n_mc,
        min_graph_prob=a.min_graph_prob,
    )  # fmt: skip
    df.to_csv(a.out, index=False)
    prefix = a.out.removesuffix(".csv")
    df_t.to_csv(prefix + "_by_t.csv", index=False)
    print(f"wrote {a.out} and {prefix}_by_t.csv")

    cols = [
        "kl_truth_model",
        "kl_truth_oracle",
        "kl_oracle_model",
        "kl_oracle_model_noflag",
    ]
    sel = df[df.k.isin(k_list)]
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(f"\n=== {MAIN_QUERY}, by experiment design and k")
        print(
            sel[sel["query"] == MAIN_QUERY]
            .groupby(["design", "k"])[cols]
            .mean()
            .round(3)
        )
        print("\n=== all 6 queries, by experiment design and k")
        print(sel.groupby(["design", "k"])[cols].mean().round(3))
        print("\n=== by query type and k")
        print(sel.groupby(["query_type", "k"])[cols].mean().round(3))

    _plot_kshot(a.out)


def _sha256(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _print_prior_summary(df) -> None:
    """Per D: how well the oracle's prior matches the task stream (true graph
    missing from the oracle's graph prior; generator max-abs resampling,
    which slightly truncates the prior relative to the oracle), design x k
    means (pairs averaged within a task first, then mean and s.e. over
    tasks) and the identity check: mean excess KL(truth) vs mean
    KL(oracle||model)."""
    import pandas as pd

    cols = [
        "kl_truth_model",
        "kl_truth_oracle",
        "kl_oracle_model",
        "excess_kl_truth",
    ]
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        for D, dd in df.groupby("D"):
            per_t = dd.drop_duplicates("task")
            missing = float((~per_t["true_graph_in_prior"]).mean())
            resampled = float((per_t["resample_attempts"] > 0).mean())
            print(
                f"\n=== D={D}: {len(per_t)} tasks; true graph missing from "
                f"the oracle's graph prior: {missing:.2%}; resampled by the "
                f"generator (resample_attempts > 0): {resampled:.2%}"
            )
            per_task = dd.groupby(["task", "design", "k"])[cols].mean()
            g = per_task.groupby(["design", "k"])
            mean, se = g.mean(), g.sem()
            print(f"\n=== D={D}: mean over tasks, by design and k (nats)")
            print(mean.round(4))
            check = pd.DataFrame({
                "excess": mean["excess_kl_truth"],
                "excess_se": se["excess_kl_truth"],
                "kl_oracle_model": mean["kl_oracle_model"],
                "kl_oracle_model_se": se["kl_oracle_model"],
            })  # fmt: skip
            print(f"\n=== D={D}: identity check, excess vs KL(oracle||model)")
            print(check.round(4))


def _run_kshot_prior(
    a: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    import json
    import os
    import time

    import pandas as pd
    import torch

    if not (a.ckpt and a.model_config and a.data_config):
        parser.error("--ckpt, --model_config and --data_config are required")

    # intra-op threads follow OMP_NUM_THREADS (eval_kshot_prior.sh sets it to
    # the Slurm CPU count; common_env.sh alone pins it to 1)
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(n_threads)

    from CITNP.datasets.create_mixed_regime_chunks import (
        _git_commit,
        load_config,
    )
    from CITNP.evaluation.experiments import run_kshot_prior
    from CITNP.evaluation.model_api import load_model
    from CITNP.evaluation.tasks import (
        default_n_obs,
        resolve_graph_prior_samples,
    )

    device = a.device or _default_device()
    data_cfg = load_config(a.data_config)
    all_D = [int(D) for D in data_cfg.num_variables]
    Ds = all_D
    if a.num_variables:
        Ds = [int(D) for D in a.num_variables.split(",")]
    if not set(Ds) <= set(all_D):
        parser.error(f"--num_variables {Ds} not a subset of {all_D}")
    k_list = sorted({int(k) for k in (a.k_list or "0,1,3,5,10").split(",")})
    n_obs = a.n_obs if a.n_obs is not None else default_n_obs(data_cfg)

    n_graphs = {
        D: resolve_graph_prior_samples(D, a.graph_prior_samples) for D in Ds
    }

    t0 = time.time()
    dfs, dfs_t = [], []
    for D in Ds:
        model = load_model(a.ckpt, a.model_config, device, num_nodes=D)
        df, df_t = run_kshot_prior(
            model, data_cfg, D=D, device=device, k_list=k_list,
            n_tasks=a.n_tasks, task_offset=a.task_offset,
            base_seed=a.base_seed, n_obs=n_obs, grid_eps=a.grid_eps,
            n_t=a.n_t, n_y=a.n_y, n_mc=a.n_mc,
            min_graph_prob=a.min_graph_prob, flag_ablation=a.flag_ablation,
            max_oracle_nodes=a.max_oracle_nodes,
            graph_prior_samples=n_graphs[D], oracle_budget=a.oracle_budget,
        )  # fmt: skip
        dfs.append(df)
        dfs_t.append(df_t)
    df = pd.concat(dfs, ignore_index=True)
    df_t = pd.concat(dfs_t, ignore_index=True)

    prefix = a.out.removesuffix(".csv")
    df.to_csv(a.out, index=False)
    df_t.to_csv(prefix + "_by_t.csv", index=False)
    tasks_df = df.drop_duplicates(["D", "task"])
    do_summary = {}
    for kind, d in tasks_df.groupby("do_kind"):
        do_summary[kind] = {
            "n_tasks": len(d),
            "param_min": float(d["do_scale_or_width"].min()),
            "param_max": float(d["do_scale_or_width"].max()),
        }
    meta = {
        "data_config": {"path": a.data_config, "sha256": _sha256(a.data_config)},
        "model_config": {
            "path": a.model_config, "sha256": _sha256(a.model_config),
        },
        "ckpt": a.ckpt,
        "git_commit": _git_commit(),
        "args": vars(a),
        "num_variables": Ds,
        "k_list": k_list,
        "n_obs": n_obs,
        "graph_prior_samples": {str(D): n for D, n in n_graphs.items()},
        "oracle_budget": a.oracle_budget,
        "torch_threads": n_threads,
        "do_dist_summary": do_summary,
        "seconds": time.time() - t0,
    }  # fmt: skip
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"wrote {a.out}, {prefix}_by_t.csv and {prefix}_meta.json")
    print(f"do-value distributions of the tasks: {do_summary}")
    _print_prior_summary(df)


# Settings that must agree across kshot-prior parts before they are merged:
# label -> key path in <out>_meta.json (missing keys read as None).
_MERGE_FIELDS = {
    "data_config sha256": ("data_config", "sha256"),
    "ckpt": ("ckpt",),
    "model_config sha256": ("model_config", "sha256"),
    "base_seed": ("args", "base_seed"),
    "k_list": ("k_list",),
    "n_mc": ("args", "n_mc"),
    "n_obs": ("n_obs",),
    "grid_eps": ("args", "grid_eps"),
    "graph_prior_samples": ("graph_prior_samples",),
    "oracle_budget": ("oracle_budget",),
}


def _meta_mismatches(metas: dict) -> dict:
    """Fields of `_MERGE_FIELDS` on which the parts' metas disagree.

    Args:
    ----------
    metas : dict
        part name -> parsed <out>_meta.json.

    Returns:
    ----------
    {field label: {part name: value}} for each differing field; empty if all
    parts agree.
    """
    import json

    out = {}
    for field, keys in _MERGE_FIELDS.items():
        vals = {}
        for part, meta in metas.items():
            v = meta
            for key in keys:
                v = v.get(key) if isinstance(v, dict) else None
            vals[part] = v
        if len({json.dumps(v, sort_keys=True) for v in vals.values()}) > 1:
            out[field] = vals
    return out


def _run_plot_kshot_prior(
    a: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    import json
    import re

    metas = {}
    for path in a.csvs:
        meta_path = path.removesuffix(".csv") + "_meta.json"
        try:
            with open(meta_path) as f:
                metas[path] = json.load(f)
        except FileNotFoundError:
            parser.error(f"missing {meta_path}")
    diff = _meta_mismatches(metas)
    if diff:
        lines = [
            f"  {field}: "
            + ", ".join(f"{part}={v!r}" for part, v in vals.items())
            for field, vals in diff.items()
        ]
        parser.error(
            "parts were run with different settings:\n" + "\n".join(lines)
        )

    import matplotlib

    matplotlib.use("Agg")
    from CITNP.evaluation.plots import plot_kshot_prior_all

    prefix = a.prefix or re.sub(r"_part\d+$", "", a.csvs[0].removesuffix(".csv"))
    plot_kshot_prior_all(a.csvs, prefix)


def _run_suite(a: argparse.Namespace) -> None:
    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.evaluation.experiments import run_scm_suite
    from CITNP.evaluation.model_api import load_model

    device = a.device or _default_device()
    model = load_model(a.ckpt, a.model_config, device)
    df = run_scm_suite(
        model, load_config(a.data_config), device=device, n_seeds=a.n_seeds,
        n_query=a.n_query, do_scale=a.do_scale, n_mc=a.n_mc,
    )  # fmt: skip
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


def _run_bank(a: argparse.Namespace) -> None:
    import pickle
    from collections import Counter

    import numpy as np

    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.evaluation.tasks import build_bank

    cfg = load_config(a.data_config)
    items = build_bank(
        cfg, per_bucket=a.per_bucket, base_seed=a.seed, n_mc=a.n_mc
    )
    with open(a.out, "wb") as f:
        pickle.dump(
            {"items": items, "data_config": a.data_config, "seed": a.seed}, f
        )

    labels = Counter(lab for it in items for lab in it["labels"])
    print(f"\nwrote {len(items)} questions to {a.out}")
    for lab, n in sorted(labels.items()):
        nll = np.mean(
            [-it["oracle_logp"].mean() for it in items if lab in it["labels"]]
        )
        print(f"  {lab:14s} n={n:4d}  oracle NLL {nll:.4f}")


def _plot_probe(csv_path: str) -> None:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    from CITNP.evaluation.plots import plot_flag_probe

    plot_flag_probe(pd.read_csv(csv_path), csv_path.removesuffix(".csv") + ".png")


def _run_probe_flag(
    a: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if a.plot_only:
        _plot_probe(a.plot_only)
        return
    if not (a.ckpt and a.model_config and a.bank):
        parser.error("--ckpt, --model_config and --bank are required")
    import pickle

    import pandas as pd

    from CITNP.evaluation.experiments import run_flag_probe
    from CITNP.evaluation.model_api import load_model

    device = a.device or _default_device()
    model = load_model(a.ckpt, a.model_config, device)
    with open(a.bank, "rb") as f:
        bank = pickle.load(f)["items"]
    df = run_flag_probe(
        model, bank, device=device, normalise=not a.no_normalise,
        max_contexts=a.max_contexts, cells_per_context=a.cells_per_context,
        n_folds=a.n_folds, seed=a.seed,
    )  # fmt: skip
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    cols = ["point", "condition", "auc", "auc_se", "bal_acc",
            "rel_change_int", "rel_change_obs"]  # fmt: skip
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df[cols].round(3).to_string(index=False))
    _plot_probe(a.out)


def _run_embedding_geometry(a: argparse.Namespace) -> None:
    import json

    from CITNP.datasets.create_mixed_regime_chunks import _git_commit
    from CITNP.evaluation.embedding_geometry import metadata_embedding_geometry
    from CITNP.evaluation.model_api import load_model

    device = a.device or _default_device()
    model = load_model(a.ckpt, a.model_config, device, num_nodes=a.num_variables)
    geometry = metadata_embedding_geometry(model.encoder, a.num_variables)
    result = {
        "ckpt": a.ckpt,
        "model_config": {
            "path": a.model_config, "sha256": _sha256(a.model_config),
        },
        "git_commit": _git_commit(),
        "geometry": geometry,
    }  # fmt: skip
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {a.out}")
    single = geometry["single_target"]
    print(f"||e_INT|| = {geometry['int_meta_embedding_norm']:.4f}  "
          f"({geometry['normalisation']})")  # fmt: skip
    for j, (norm, cosine, normalised_cosine) in enumerate(zip(
        geometry["index_embedding_norms"],
        geometry["cosine_int_meta_vs_index"],
        single["normalised_cosine_to_observational"],
    )):  # fmt: skip
        print(f"  j={j}: ||e_j|| = {norm:.4f}, cos(e_INT, e_j) = {cosine:+.3f}, "
              f"cos(N(e_INT + e_j), N(e_INT)) = {normalised_cosine:+.4f}")  # fmt: skip
    print("  min distance between N(e_INT + e_j): "
          f"{single['min_pairwise_normalised_distance']}")  # fmt: skip


def _load_diagnostic_inputs(a: argparse.Namespace):
    """(prior config, D, device, loaded model) of a diagnostic command."""
    from CITNP.evaluation.model_api import load_model
    from CITNP.evaluation.tasks import load_prior_config

    data_cfg = load_prior_config(a.data_config)
    D = a.num_variables or min(int(d) for d in data_cfg.num_variables)
    device = a.device or _default_device()
    model = load_model(a.ckpt, a.model_config, device, num_nodes=D)
    return data_cfg, D, device, model


def _write_diagnostic_meta(
    a: argparse.Namespace, D: int, seconds: float, **extra
) -> str:
    """Write <out>_meta.json (inputs, git commit, arguments) next to the
    results of a diagnostic command; returns its path."""
    import json

    from CITNP.datasets.create_mixed_regime_chunks import _git_commit

    meta = {
        "data_config": {"path": a.data_config, "sha256": _sha256(a.data_config)},
        "model_config": {
            "path": a.model_config, "sha256": _sha256(a.model_config),
        },
        "ckpt": a.ckpt,
        "git_commit": _git_commit(),
        "args": vars(a),
        "num_variables": D,
        "seconds": seconds,
        **extra,
    }  # fmt: skip
    path = a.out.removesuffix(".csv") + "_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def _run_probe_intervention(a: argparse.Namespace) -> None:
    import time

    import pandas as pd

    from CITNP.evaluation.intervention_diagnostics import run_intervention_probes
    from CITNP.evaluation.model_api import probe_points, uses_metadata_node

    t0 = time.time()
    data_cfg, D, device, model = _load_diagnostic_inputs(a)
    df = run_intervention_probes(
        model, data_cfg, D=D, device=device, n_tasks=a.n_tasks,
        task_offset=a.task_offset, base_seed=a.base_seed,
        rows_per_task=a.rows_per_task, n_folds=a.n_folds, l2=a.l2,
        max_iter=a.max_iter, seed=a.seed, normalise=not a.no_normalise,
    )  # fmt: skip
    df.to_csv(a.out, index=False)
    meta = _write_diagnostic_meta(
        a, D, time.time() - t0, probe_points=probe_points(model),
        metadata_column=uses_metadata_node(model),
    )  # fmt: skip
    print(f"wrote {a.out} and {meta}")
    table = df.pivot_table(
        index=["probe", "depth", "point"], columns="labels",
        values="balanced_accuracy",
    )  # fmt: skip
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print("\n=== held-out balanced accuracy (true vs shuffled labels)")
        print(table.round(3))


def _run_context_edits(a: argparse.Namespace) -> None:
    import os
    import time

    import pandas as pd
    import torch

    from CITNP.evaluation.intervention_diagnostics import (
        run_context_edits,
        summarise_context_edits,
    )

    # as kshot-prior: the exact oracle dominates the cost
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    t0 = time.time()
    data_cfg, D, device, model = _load_diagnostic_inputs(a)
    df = run_context_edits(
        model, data_cfg, D=D, device=device, n_tasks=a.n_tasks,
        task_offset=a.task_offset, base_seed=a.base_seed, n_mc=a.n_mc,
        n_y=a.n_y, min_graph_prob=a.min_graph_prob,
        max_query_points=a.max_query_points, max_edit_steps=a.max_edit_steps,
        seed=a.seed, normalise=not a.no_normalise,
        keep_original_normalisation=a.keep_original_normalisation,
        max_oracle_nodes=a.max_oracle_nodes,
        graph_prior_samples=a.graph_prior_samples,
    )  # fmt: skip
    summary = summarise_context_edits(df)
    prefix = a.out.removesuffix(".csv")
    df.to_csv(a.out, index=False)
    summary.to_csv(prefix + "_summary.csv", index=False)
    meta = _write_diagnostic_meta(a, D, time.time() - t0)
    print(f"wrote {a.out}, {prefix}_summary.csv and {meta}")
    cols = ["edit", "identifiable_from_observational",
            "treatment_intervened_in_context", "num_queries",
            "mean_shift_ratio", "mean_shift_correlation",
            "mean_shift_sign_agreement", "kl_ratio", "kl_correlation"]  # fmt: skip
    if summary.empty:
        print("no task had an intervened context row")
        return
    overall = summary[summary["num_edited_rows"] == "all"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print("\n=== label sensitivity, model vs oracle (all numbers of "
              "edited rows pooled)")  # fmt: skip
        print(overall[cols].round(3).to_string(index=False))


def _run_plot_monitoring(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from CITNP.evaluation.plots import plot_monitoring

    plot_monitoring(a.run_dirs, a.out)


COMMANDS = {
    "kshot": (
        _kshot_parser,
        "k-shot evaluation on the confounder SCM (+ figures)",
    ),
    "kshot-prior": (
        _kshot_prior_parser,
        "k-shot evaluation on tasks drawn from the training prior",
    ),
    "scm-suite": (
        _suite_parser,
        "fixed SCMs x context designs vs truth and oracle",
    ),
    "bank": (_bank_parser, "build the fixed oracle bank used during training"),
    "calibrate": (
        _calibrate_parser,
        "oracle simulation-based calibration: PIT and MEC posterior",
    ),
    "plot-kshot": (_plot_kshot_parser, "all k-shot figures from saved CSVs"),
    "plot-kshot-prior": (
        _plot_kshot_prior_parser,
        "figures from kshot-prior CSV parts (same settings only)",
    ),
    "plot-monitoring": (
        _plot_monitoring_parser,
        "figures from oracle_gap / train_stats CSVs",
    ),
    "probe-flag": (
        _probe_flag_parser,
        "linear probes: is the intervention flag still present at each depth?",
    ),
    "embedding-geometry": (
        _embedding_geometry_parser,
        "extra_node: norms and cosines of the metadata embeddings (no data)",
    ),
    "probe-intervention": (
        _probe_intervention_parser,
        "per-layer linear probes: is a row / cell intervened (any encoding)?",
    ),
    "context-edits": (
        _context_edits_parser,
        "label sensitivity: edit the intervention mask, model vs exact oracle",
    ),
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch `python -m CITNP.evaluation <command> ...`."""
    ap = argparse.ArgumentParser(prog="python -m CITNP.evaluation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="command", required=True)
    parsers = {}
    for name, (build, help_) in COMMANDS.items():
        parsers[name] = sub.add_parser(name, help=help_)
        build(parsers[name])
    a = ap.parse_args(argv)
    if a.command == "kshot":
        _run_kshot(a, parsers["kshot"])
    elif a.command == "kshot-prior":
        _run_kshot_prior(a, parsers["kshot-prior"])
    elif a.command == "scm-suite":
        _run_suite(a)
    elif a.command == "bank":
        _run_bank(a)
    elif a.command == "calibrate":
        from CITNP.evaluation.oracle_calibration import run

        run(a)
    elif a.command == "plot-kshot":
        _plot_kshot(a.csv)
    elif a.command == "plot-kshot-prior":
        _run_plot_kshot_prior(a, parsers["plot-kshot-prior"])
    elif a.command == "probe-flag":
        _run_probe_flag(a, parsers["probe-flag"])
    elif a.command == "embedding-geometry":
        _run_embedding_geometry(a)
    elif a.command == "probe-intervention":
        _run_probe_intervention(a)
    elif a.command == "context-edits":
        _run_context_edits(a)
    else:
        _run_plot_monitoring(a)


def _command(name: str):
    """Entry point running one subcommand on sys.argv[1:] (for old scripts)."""

    def run(argv: list[str] | None = None) -> None:
        main([name, *(sys.argv[1:] if argv is None else argv)])

    run.__name__ = run.__qualname__ = name.replace("-", "_") + "_cli"
    return run


kshot_cli = _command("kshot")
kshot_prior_cli = _command("kshot-prior")
scm_suite_cli = _command("scm-suite")
bank_cli = _command("bank")
plot_kshot_cli = _command("plot-kshot")
plot_monitoring_cli = _command("plot-monitoring")
probe_flag_cli = _command("probe-flag")


if __name__ == "__main__":
    main()
