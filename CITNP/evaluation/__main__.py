"""Command-line entry point for model evaluation (argument parsing only).

    python -m CITNP.evaluation kshot --ckpt ... --model_config ... --data_config ... \
           [--k_list 0,1,3,5,10,20,50] [--n_seeds 100] [--random_weights] --out kshot.csv
    python -m CITNP.evaluation scm-suite --ckpt ... --model_config ... \
           --data_config ... --out suite.csv
    python -m CITNP.evaluation bank --data_config ... --out bank.pkl
    python -m CITNP.evaluation plot-kshot kshot.csv     # all k-shot figures, no GPU
    python -m CITNP.evaluation plot-monitoring RUN_DIR [RUN_DIR ...] --out monitoring.png

See CITNP.evaluation.experiments for what `kshot` and `scm-suite` compute,
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
def _kshot_parser(p: argparse.ArgumentParser) -> None:
    _model_args(p, required=False)
    p.add_argument("--plot_only", default=None, help="CSV from a previous run: only plot")
    p.add_argument("--n_seeds", type=int, default=100)
    p.add_argument("--n_obs", type=int, default=128)
    p.add_argument("--k_max", type=int, default=10)
    p.add_argument("--k_list", default=None,
                   help="comma-separated numbers of experiments, e.g. 0,1,3,5,10; "
                        "overrides --k_max (default: 0..k_max)")  # fmt: skip
    p.add_argument("--random_weights", action="store_true",
                   help="fresh confounder-SCM weights per seed (same graph)")  # fmt: skip
    p.add_argument("--do_scale", type=float, default=1.5,
                   help="sd of do-values in the context's interventional rows")  # fmt: skip
    p.add_argument("--t_max", type=float, default=3.0, help="query grid is [-t_max, t_max]")
    p.add_argument("--n_t", type=int, default=13, help="query grid points")
    p.add_argument("--n_y", type=int, default=64, help="MC samples per grid point for KLs")
    p.add_argument("--n_mc", type=int, default=200, help="oracle parameter draws per DAG")
    p.add_argument("--min_graph_prob", type=float, default=1e-4)
    p.add_argument("--out", default="kshot.csv")
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")


def _suite_parser(p: argparse.ArgumentParser) -> None:
    _model_args(p, required=True)
    p.add_argument("--n_seeds", type=int, default=30)
    p.add_argument("--n_query", type=int, default=256)
    p.add_argument("--do_scale", type=float, default=1.5)
    p.add_argument("--n_mc", type=int, default=1000)
    p.add_argument("--out", default="results_d3_oracle.csv")
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")


def _bank_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per_bucket", type=int, default=150)
    p.add_argument("--n_mc", type=int, default=2000)
    p.add_argument("--seed", type=int, default=4242)


def _plot_kshot_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("csv", help="<prefix>.csv written by kshot (reads <prefix>_by_t.csv too)")


def _plot_monitoring_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default="monitoring.png")


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

    cols = ["kl_truth_model", "kl_truth_oracle", "kl_oracle_model", "kl_oracle_model_noflag"]
    sel = df[df.k.isin(k_list)]
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(f"\n=== {MAIN_QUERY}, by experiment design and k")
        print(sel[sel["query"] == MAIN_QUERY].groupby(["design", "k"])[cols].mean().round(3))
        print("\n=== all 6 queries, by experiment design and k")
        print(sel.groupby(["design", "k"])[cols].mean().round(3))
        print("\n=== by query type and k")
        print(sel.groupby(["query_type", "k"])[cols].mean().round(3))

    _plot_kshot(a.out)


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
    items = build_bank(cfg, per_bucket=a.per_bucket, base_seed=a.seed, n_mc=a.n_mc)
    with open(a.out, "wb") as f:
        pickle.dump({"items": items, "data_config": a.data_config, "seed": a.seed}, f)

    labels = Counter(lab for it in items for lab in it["labels"])
    print(f"\nwrote {len(items)} questions to {a.out}")
    for lab, n in sorted(labels.items()):
        nll = np.mean([-it["oracle_logp"].mean() for it in items if lab in it["labels"]])
        print(f"  {lab:14s} n={n:4d}  oracle NLL {nll:.4f}")


def _run_plot_monitoring(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from CITNP.evaluation.plots import plot_monitoring

    plot_monitoring(a.run_dirs, a.out)


COMMANDS = {
    "kshot": (_kshot_parser, "k-shot evaluation on the confounder SCM (+ figures)"),
    "scm-suite": (_suite_parser, "fixed SCMs x context designs vs truth and oracle"),
    "bank": (_bank_parser, "build the fixed oracle bank used during training"),
    "plot-kshot": (_plot_kshot_parser, "all k-shot figures from saved CSVs"),
    "plot-monitoring": (_plot_monitoring_parser, "figures from oracle_gap / train_stats CSVs"),
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
    elif a.command == "scm-suite":
        _run_suite(a)
    elif a.command == "bank":
        _run_bank(a)
    elif a.command == "plot-kshot":
        _plot_kshot(a.csv)
    else:
        _run_plot_monitoring(a)


def _command(name: str):
    """Entry point running one subcommand on sys.argv[1:] (for old scripts)."""

    def run(argv: list[str] | None = None) -> None:
        main([name, *(sys.argv[1:] if argv is None else argv)])

    run.__name__ = run.__qualname__ = name.replace("-", "_") + "_cli"
    return run


kshot_cli = _command("kshot")
scm_suite_cli = _command("scm-suite")
bank_cli = _command("bank")
plot_kshot_cli = _command("plot-kshot")
plot_monitoring_cli = _command("plot-monitoring")


if __name__ == "__main__":
    main()
