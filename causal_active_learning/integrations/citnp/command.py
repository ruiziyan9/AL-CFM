"""Command-line entry points of the CITNP integration:

    python -m causal_active_learning.integrations.citnp run --ckpt ... \
           --model_config ... --data_config ... --environment example \
           --fixed_graph_edges 0->1,1->2,2->4,0->3,3->4,0->2 --treatment 2 \
           --outcome 4 [--strategies ...] [--experiment_budget 10] \
           [--seeds 0,1,2,3,4] --out al
    python -m causal_active_learning.integrations.citnp plot al
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


def _sha256(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


_ACTIVE_LEARNING_INT_SETTINGS = (
    "rows_per_experiment", "num_fantasies", "final_num_fantasies",
    "quadrature_points", "kl_monte_carlo_samples", "inference_batch_size",
    "bayesian_optimisation_initial_points", "bayesian_optimisation_iterations",
    "acquisition_grid_points", "gaussian_process_restarts",
)  # fmt: skip
_ACTIVE_LEARNING_FLOAT_SETTINGS = (
    "value_domain_half_width", "ucb_exploration_weight",
)  # fmt: skip


def run_parser(p: argparse.ArgumentParser) -> None:
    # Loop settings default to None = ActiveLearningConfig's default (the
    # package is imported only when the command runs).
    _model_args(p, required=False)
    p.add_argument("--environment", choices=["example", "prior"],
                   default="example",
                   help="'example': one fixed model (--example_file, or drawn "
                        "on --fixed_graph_edges from --data_config's prior); "
                        "'prior': tasks of the training prior")  # fmt: skip
    p.add_argument("--example_file", default=None,
                   help="YAML example model (as in kshot-example)")  # fmt: skip
    p.add_argument("--fixed_graph_edges", default=None,
                   help="example: the DAG to draw the model on (0-based "
                        "'source->target' edges); prior: every task uses this "
                        "DAG")  # fmt: skip
    p.add_argument("--example_seed", type=int, default=9101,
                   help="seed for drawing the example model from the prior")  # fmt: skip
    p.add_argument("--n_tasks", type=int, default=50,
                   help="prior: number of tasks")  # fmt: skip
    p.add_argument("--task_offset", type=int, default=0,
                   help="prior: first task index")  # fmt: skip
    p.add_argument("--treatment", type=int, default=None,
                   help="0-based treatment node T (required for 'example'; "
                        "for 'prior' default: a random pair per task)")  # fmt: skip
    p.add_argument("--outcome", type=int, default=None,
                   help="0-based outcome node Y")  # fmt: skip
    p.add_argument("--n_obs", type=int, default=None,
                   help="initial observational rows (default: from the data "
                        "config, as in kshot-prior)")  # fmt: skip
    p.add_argument("--strategies", default=None,
                   help="comma-separated subset of expected_information_gain, "
                        "random, observational, treatment, "
                        "exhaustive_true_gain (default: all)")  # fmt: skip
    p.add_argument("--value_optimisers", default="bayesian_optimisation",
                   help="comma-separated: bayesian_optimisation, grid (for the "
                        "strategies with an inner level)")  # fmt: skip
    p.add_argument("--seeds", default="0",
                   help="comma-separated seeds, e.g. 0,1,2,3,4")  # fmt: skip
    p.add_argument("--base_seed", type=int, default=9101)
    p.add_argument("--n_t", type=int, default=13, help="query grid points")
    p.add_argument("--grid_eps", type=float, default=0.02,
                   help="query grid: quantiles [eps, 1 - eps] of the task's "
                        "do-value distribution")  # fmt: skip
    p.add_argument("--experiment_budget", type=int, default=10)
    for name in _ACTIVE_LEARNING_INT_SETTINGS:
        p.add_argument(f"--{name}", type=int, default=None,
                       help="default: as in ActiveLearningConfig")  # fmt: skip
    for name in _ACTIVE_LEARNING_FLOAT_SETTINGS:
        p.add_argument(f"--{name}", type=float, default=None,
                       help="default: as in ActiveLearningConfig")  # fmt: skip
    p.add_argument("--fantasy_row_sampling", default="autoregressive",
                   choices=["autoregressive", "independent"],
                   help="independent: for comparison only")  # fmt: skip
    p.add_argument("--kl_estimator", default=None,
                   choices=["quadrature", "monte_carlo"])  # fmt: skip
    p.add_argument("--value_domain", default=None,
                   choices=["observational_range", "prior_range"])  # fmt: skip
    p.add_argument("--grid_standardised_values", default=None,
                   help="comma-separated, default -2,-1,0,1,2")  # fmt: skip
    p.add_argument("--include_treatment_candidate", action="store_true",
                   help="exclude_query_nodes=False: T is a candidate too")  # fmt: skip
    p.add_argument("--include_observational_candidate", action="store_true")
    p.add_argument("--dense_value_diagnostic", action="store_true",
                   help="fixed example only (expensive): also evaluate EIG on "
                        "a dense value grid per node")  # fmt: skip
    p.add_argument("--dense_value_diagnostic_steps", default="1",
                   help="steps (1-based) of the dense diagnostic")  # fmt: skip
    p.add_argument("--dense_value_diagnostic_points", type=int, default=41)
    p.add_argument("--out", default="active_learning",
                   help="output prefix: <out>_steps.csv, <out>_scores.csv, "
                        "<out>_bayesian_optimisation_trace.csv, figures")  # fmt: skip
    p.add_argument("--device", default=None,
                   help="default: cuda if available, else cpu")  # fmt: skip


def plot_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("prefix", help="the --out prefix of an active-learning run")
    p.add_argument("--surrogate_step", type=int, default=1)


def run_command(a: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import json
    import os
    import time
    from dataclasses import asdict

    import torch

    if not (a.ckpt and a.model_config):
        parser.error("--ckpt and --model_config are required")
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(n_threads)

    from causal_active_learning.config import (
        ACQUISITIONS,
        ActiveLearningConfig,
    )
    from causal_active_learning.integrations.citnp.environment import (
        LinearSCMEnvironment,
    )
    from causal_active_learning.integrations.citnp.experiment import (
        ActiveLearningTask,
        make_strategies,
        prior_tasks,
        run_active_learning_experiment,
    )
    from causal_active_learning.integrations.citnp.model import (
        ModelApiPredictiveModel,
    )
    from causal_active_learning.integrations.citnp.support import (
        training_support_from_data_config,
    )
    from CITNP.datasets.create_mixed_regime_chunks import (
        _git_commit,
        load_config,
    )
    from CITNP.evaluation.example_model import (
        draw_example_model,
        example_model_to_dict,
        load_example_model,
        write_example_model,
    )
    from CITNP.evaluation.model_api import load_model
    from CITNP.evaluation.tasks import default_n_obs, parse_graph_edges
    from CITNP.prior_diagnostics.stats import question_class

    data_cfg = load_config(a.data_config) if a.data_config else None
    prefix = a.out.removesuffix(".csv")
    strategies_given = (
        list(ACQUISITIONS) if a.strategies is None
        else [s.strip() for s in a.strategies.split(",") if s.strip()]
    )  # fmt: skip
    unknown = [s for s in strategies_given if s not in ACQUISITIONS]
    if unknown:
        parser.error(f"--strategies: unknown {unknown}; valid: {ACQUISITIONS}")
    optimisers = [s.strip() for s in a.value_optimisers.split(",") if s.strip()]
    settings = {
        name: getattr(a, name)
        for name in (
            *_ACTIVE_LEARNING_INT_SETTINGS, *_ACTIVE_LEARNING_FLOAT_SETTINGS,
            "kl_estimator", "value_domain",
        )
        if getattr(a, name) is not None
    }  # fmt: skip
    if a.grid_standardised_values is not None:
        settings["grid_standardised_values"] = tuple(
            float(u) for u in a.grid_standardised_values.split(",")
        )
    try:
        base = ActiveLearningConfig(
            experiment_budget=a.experiment_budget,
            exclude_query_nodes=not a.include_treatment_candidate,
            include_observational_candidate=a.include_observational_candidate,
            **settings,
        )
        strategies = make_strategies(base, strategies_given, optimisers)
    except ValueError as error:
        parser.error(str(error))
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    example_meta = None
    if a.environment == "example":
        if a.example_file:
            example = load_example_model(a.example_file)
            example_path = a.example_file
        else:
            if not (a.fixed_graph_edges and data_cfg is not None):
                parser.error(
                    "--environment example needs --example_file, or "
                    "--fixed_graph_edges with --data_config"
                )
            all_D = [int(D) for D in data_cfg.num_variables]
            if len(all_D) != 1:
                parser.error("the data config must have a single D")
            try:
                graph = parse_graph_edges(a.fixed_graph_edges, all_D[0])
            except ValueError as error:
                parser.error(f"--fixed_graph_edges: {error}")
            example = draw_example_model(
                data_cfg, graph, a.example_seed, data_config_path=a.data_config
            )
            example_path = prefix + "_example.yaml"
            write_example_model(example_path, example)
            print(
                f"drew the example model from the prior, saved to {example_path}"
            )
        D = example.num_variables
        T, Y = a.treatment, a.outcome
        if T is None or Y is None or not (0 <= T < D and 0 <= Y < D and T != Y):
            parser.error(
                "--environment example needs distinct 0-based --treatment and "
                f"--outcome below D = {D}"
            )
        tasks = [
            ActiveLearningTask(
                0, LinearSCMEnvironment.from_example_model(example), T, Y,
                question_class(example.graph, T, Y),
            )
        ]  # fmt: skip
        example_meta = {
            "example_file": example_path,
            "example_model": example_model_to_dict(example),
        }
    else:
        if a.dense_value_diagnostic:
            parser.error("--dense_value_diagnostic is for the fixed example only")
        if data_cfg is None:
            parser.error("--environment prior needs --data_config")
        all_D = [int(D) for D in data_cfg.num_variables]
        if len(all_D) != 1:
            parser.error("the data config must have a single D")
        D = all_D[0]
        if (a.treatment is None) != (a.outcome is None):
            parser.error("give both --treatment and --outcome, or neither")
        fixed_graph = None
        if a.fixed_graph_edges:
            try:
                fixed_graph = parse_graph_edges(a.fixed_graph_edges, D)
            except ValueError as error:
                parser.error(f"--fixed_graph_edges: {error}")
        tasks = prior_tasks(
            data_cfg, D, range(a.task_offset, a.task_offset + a.n_tasks),
            a.base_seed, a.treatment, a.outcome, fixed_graph,
        )  # fmt: skip

    if a.n_obs is not None:
        n_obs = a.n_obs
    elif data_cfg is not None:
        n_obs = default_n_obs(data_cfg)
    else:
        parser.error("pass --n_obs (there is no data config to take it from)")
    support = None
    if data_cfg is not None:
        support = training_support_from_data_config(data_cfg, D)
        print(f"training support of the data config: {support}")
    device = a.device or _default_device()
    model = load_model(a.ckpt, a.model_config, device, num_nodes=D)
    try:
        predictive_model = ModelApiPredictiveModel(
            model,
            device,
            fantasy_row_sampling=a.fantasy_row_sampling,
            inference_batch_size=base.inference_batch_size,
        )
    except ValueError as error:
        parser.error(str(error))
    dense_steps = (
        [int(s) for s in a.dense_value_diagnostic_steps.split(",") if s.strip()]
        if a.dense_value_diagnostic else []
    )  # fmt: skip

    t0 = time.time()
    tables = run_active_learning_experiment(
        predictive_model, tasks, strategies, seeds, n_obs,
        base_seed=a.base_seed, training_support=support,
        query_grid_points=a.n_t, grid_eps=a.grid_eps,
        dense_value_diagnostic_steps=dense_steps,
        dense_value_diagnostic_points=a.dense_value_diagnostic_points,
        progress=print,
    )  # fmt: skip
    outputs = {
        "steps": tables.steps, "scores": tables.scores,
        "bayesian_optimisation_trace": tables.trace,
        "surrogate": tables.surrogate,
        "dense_value_diagnostic": tables.dense,
    }  # fmt: skip
    for name, df in outputs.items():
        df.to_csv(f"{prefix}_{name}.csv", index=False)

    def _config_meta(path: str) -> dict:
        from CITNP.utils.config_files import config_fingerprint

        return {
            "path": path, "sha256": _sha256(path),
            "fingerprint": config_fingerprint(path),
        }  # fmt: skip

    meta = {
        "model_config": _config_meta(a.model_config),
        "data_config": _config_meta(a.data_config) if a.data_config else None,
        "ckpt": a.ckpt,
        "environment": a.environment,
        **(example_meta or {}),
        "tasks": [
            {"task": t.task, "treatment": t.treatment, "outcome": t.outcome,
             "relation": t.relation}
            for t in tasks
        ],
        "strategies": {s.label: asdict(s.config) for s in strategies},
        "seeds": seeds,
        "n_obs": n_obs,
        "training_support": None if support is None else asdict(support),
        "git_commit": _git_commit(),
        "args": vars(a),
        "torch_threads": n_threads,
        "seconds": time.time() - t0,
    }  # fmt: skip
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"wrote {prefix}_*.csv and {prefix}_meta.json")
    if not tables.steps.empty:
        summary = tables.steps.groupby(["strategy", "step"])[
            ["kl_to_truth_after", "realised_information_gain"]
        ].mean()
        print(summary.round(4).to_string())
    plot_command(prefix)


def plot_command(prefix: str, surrogate_step: int = 1) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from causal_active_learning.evaluation.plots import (
        plot_active_learning_all,
    )

    for path in plot_active_learning_all(prefix, surrogate_step):
        print(f"wrote {path}")


COMMANDS = {
    "run": (run_parser, "run the active learning loop for several strategies"),
    "plot": (plot_parser, "redraw the figures of a run from its CSVs"),
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch `python -m causal_active_learning.integrations.citnp ...`."""
    ap = argparse.ArgumentParser(
        prog="python -m causal_active_learning.integrations.citnp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    parsers = {}
    for name, (build, help_) in COMMANDS.items():
        parsers[name] = sub.add_parser(name, help=help_)
        build(parsers[name])
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.command == "run":
        run_command(a, parsers["run"])
    else:
        plot_command(a.prefix, a.surrogate_step)
