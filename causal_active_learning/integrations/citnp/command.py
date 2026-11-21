"""Command-line entry points of the CITNP integration:

    python -m causal_active_learning.integrations.citnp run \\
        --ckpt RUN/checkpoint.pt --model_config RUN/resolved_model_config.yaml \\
        --data_config <data yaml> --experiment <experiment yaml> \\
        --out PREFIX [--seed 0] [--task_indices 0,1] [--restart_indices 0,1,2]
    python -m causal_active_learning.integrations.citnp merge \\
        --inputs PREFIX_part1 PREFIX_part2 ... --out PREFIX
    python -m causal_active_learning.integrations.citnp plot-panels \\
        --summaries PREFIX_a_summary.csv PREFIX_b_summary.csv \\
        --experiments <yaml a> <yaml b> \\
        --titles "All nodes intervenable" "X3 not intervenable" --out FILE
    python -m causal_active_learning.integrations.citnp plot PREFIX

`run` executes one experiment config (one setting). Without --experiment, the
ad hoc arguments describe a quick run: --environment example (one fixed model,
from --example_file or drawn on --fixed_graph_edges from the data config's
prior; restarts = --seeds) or --environment prior (--n_tasks tasks from
--task_offset); the psi values are then the k-shot quantile grid of each
task's do-value distribution. --experiment and the ad hoc arguments must not
be combined. Loop settings given on the command line override the
experiment's `active_learning` section.
"""

import argparse
import sys

AD_HOC_ARGUMENTS = (
    "environment",
    "example_file",
    "fixed_graph_edges",
    "n_tasks",
    "task_offset",
    "treatment",
    "outcome",
    "n_obs",
    "strategies",
    "seeds",
    "intervenable_nodes",
)
INT_SETTINGS = (
    "acquisition_budget",
    "rows_per_acquisition",
    "num_fantasies",
    "final_num_fantasies",
    "quadrature_points",
    "kl_monte_carlo_samples",
    "inference_batch_size",
    "grid_num_values",
    "bayesian_optimisation_initial_points",
    "bayesian_optimisation_iterations",
    "acquisition_grid_points",
    "gaussian_process_restarts",
)
FLOAT_SETTINGS = (
    "value_domain_half_width",
    "ucb_exploration_weight",
    "fixed_intervention_value",
    "observational_cost",
    "intervention_cost",
)
CHOICE_SETTINGS = {
    "fantasy_noise": ("independent", "shared"),
    "kl_estimator": ("quadrature", "monte_carlo"),
    "value_domain": ("observational_range", "training_range", "fixed_range"),
    "gaussian_process_noise": ("auto", "standard_error", "fixed", "fitted"),
}


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _sha256(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    return [int(v) for v in text.split(",") if v.strip()]


def _floats(text: str | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    return tuple(float(v) for v in text.split(",") if v.strip())


def run_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument(
        "--experiment", default=None, help="experiment YAML (one setting)"
    )
    p.add_argument("--seed", type=int, default=0, help="experiment seed")
    p.add_argument(
        "--task_indices",
        default=None,
        help="comma-separated subset of the experiment's tasks",
    )
    p.add_argument(
        "--restart_indices",
        default=None,
        help="comma-separated subset of the restarts",
    )
    p.add_argument(
        "--surrogate_steps",
        default="1",
        help="steps whose GP posteriors are written",
    )
    p.add_argument(
        "--dense_value_diagnostic",
        action="store_true",
        help="fixed graph only (expensive): EIG on a dense value "
        "grid per node, both noise modes",
    )
    p.add_argument("--dense_value_diagnostic_steps", default="1")
    p.add_argument("--dense_value_diagnostic_points", type=int, default=41)
    p.add_argument(
        "--fantasy_row_sampling",
        default="autoregressive",
        choices=["autoregressive", "independent"],
        help="independent: for comparison only",
    )
    p.add_argument(
        "--minimum_observational_rows",
        type=int,
        default=10,
        help="fewest observational rows the model accepts in a "
        "context (10, as in training; lower only knowingly)",
    )
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )
    # ad hoc runs (without --experiment); default None = not given
    p.add_argument("--environment", choices=["example", "prior"], default=None)
    p.add_argument("--example_file", default=None)
    p.add_argument(
        "--fixed_graph_edges", default=None, help="0-based 'source->target' edges"
    )
    p.add_argument("--example_seed", type=int, default=9101)
    p.add_argument("--n_tasks", type=int, default=None)
    p.add_argument("--task_offset", type=int, default=None)
    p.add_argument("--treatment", type=int, default=None)
    p.add_argument("--outcome", type=int, default=None)
    p.add_argument("--n_obs", type=int, default=None)
    p.add_argument(
        "--strategies", default=None, help="comma-separated (default: all)"
    )
    p.add_argument(
        "--seeds", default=None, help="ad hoc restarts, e.g. 0,1,2,3,4"
    )
    p.add_argument(
        "--intervenable_nodes",
        default=None,
        help="comma-separated 0-based nodes (default: all)",
    )
    # loop settings (override the experiment's active_learning section)
    p.add_argument(
        "--value_search",
        default=None,
        help="bayesian_optimisation, grid, or both comma-separated",
    )
    for name in INT_SETTINGS:
        p.add_argument(f"--{name}", type=int, default=None)
    for name in FLOAT_SETTINGS:
        p.add_argument(f"--{name}", type=float, default=None)
    for name, choices in CHOICE_SETTINGS.items():
        p.add_argument(f"--{name}", choices=choices, default=None)
    p.add_argument("--value_domain_fixed_range", default=None, help="low,high")
    p.add_argument("--random_value_range", default=None, help="low,high")
    p.add_argument("--grid_standardised_values", default=None)
    p.add_argument("--no_observational_candidate", action="store_true")


def merge_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--inputs", nargs="+", required=True, help="part prefixes")
    p.add_argument("--out", required=True, help="merged prefix")


def plot_panels_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--titles", nargs="+", default=None)
    p.add_argument("--all_bands", action="store_true")
    p.add_argument("--out", required=True)


def plot_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("prefix", help="the --out prefix of a run")


def _overrides(a: argparse.Namespace) -> dict:
    settings = {
        name: getattr(a, name)
        for name in (*INT_SETTINGS, *FLOAT_SETTINGS, *CHOICE_SETTINGS)
        if getattr(a, name) is not None
    }
    for name in (
        "value_domain_fixed_range",
        "random_value_range",
        "grid_standardised_values",
    ):
        if getattr(a, name) is not None:
            settings[name] = _floats(getattr(a, name))
    if a.no_observational_candidate:
        settings["include_observational_candidate"] = False
    if a.intervenable_nodes is not None:
        settings["intervenable_nodes"] = tuple(_ints(a.intervenable_nodes) or ())
    return settings


def run_command(a: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import json
    import os
    import time
    from dataclasses import asdict

    import torch

    from causal_active_learning.config import STRATEGIES, ActiveLearningConfig
    from causal_active_learning.evaluation.summaries import summarise
    from causal_active_learning.integrations.citnp.environment import (
        LinearSCMEnvironment,
    )
    from causal_active_learning.integrations.citnp.experiment import (
        ActiveLearningTask,
        ExperimentSpec,
        build_tasks,
        load_experiment,
        make_strategies,
        query_treatment_grid,
        run_experiment,
    )
    from causal_active_learning.integrations.citnp.model import (
        CITNPPredictiveModel,
    )
    from causal_active_learning.integrations.citnp.support import (
        training_support_from_data_config,
    )
    from causal_active_learning.acquisitions import InterventionalQuery
    from CITNP.datasets.create_mixed_regime_chunks import _git_commit, load_config
    from CITNP.evaluation.example_model import (
        draw_example_model,
        example_model_to_dict,
        load_example_model,
        write_example_model,
    )
    from CITNP.evaluation.model_api import load_model
    from CITNP.evaluation.tasks import default_n_obs, parse_graph_edges
    from CITNP.prior_diagnostics.stats import question_class
    from CITNP.utils.config_files import config_fingerprint

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    data_cfg = load_config(a.data_config)
    prefix = a.out.removesuffix(".csv")
    overrides = _overrides(a)
    value_searches = None
    if a.value_search is not None:
        value_searches = [
            s.strip() for s in a.value_search.split(",") if s.strip()
        ]
    example_meta = None
    spec: ExperimentSpec | None = None
    given_ad_hoc = [n for n in AD_HOC_ARGUMENTS if getattr(a, n) is not None]
    try:
        if a.experiment is not None:
            if given_ad_hoc:
                parser.error(
                    f"--experiment cannot be combined with {given_ad_hoc}"
                )
            spec = load_experiment(a.experiment)
            setting = spec.name
            task_indices = _ints(a.task_indices) or list(range(spec.num_tasks))
            restarts = _ints(a.restart_indices) or list(
                range(spec.restarts_per_task)
            )
            tasks = build_tasks(spec, data_cfg, a.seed, task_indices)
            initial_rows = spec.initial_observational_rows
            names = spec.strategies
            configs = lambda name: spec.config(name, **overrides)  # noqa: E731
        else:
            if a.environment is None:
                parser.error(
                    "give --experiment, or --environment for an ad hoc run"
                )
            setting = a.environment
            names = (
                list(STRATEGIES)
                if a.strategies is None
                else [s.strip() for s in a.strategies.split(",") if s.strip()]
            )
            restarts = _ints(a.seeds) or [0]
            budget = overrides.pop("acquisition_budget", 10)
            configs = lambda name: ActiveLearningConfig(  # noqa: E731
                acquisition_budget=budget, strategy=name, **overrides
            )
            if a.environment == "example":
                if a.example_file:
                    example = load_example_model(a.example_file)
                    example_path = a.example_file
                elif a.fixed_graph_edges:
                    D = int(data_cfg.num_variables[0])
                    graph = parse_graph_edges(a.fixed_graph_edges, D)
                    example = draw_example_model(
                        data_cfg,
                        graph,
                        a.example_seed,
                        data_config_path=a.data_config,
                    )
                    example_path = prefix + "_example.yaml"
                    write_example_model(example_path, example)
                else:
                    parser.error(
                        "--environment example needs --example_file "
                        "or --fixed_graph_edges"
                    )
                if a.treatment is None or a.outcome is None:
                    parser.error(
                        "--environment example needs --treatment and --outcome"
                    )
                env = LinearSCMEnvironment.from_example_model(example)
                assert env.do_value_distribution is not None
                psi = torch.as_tensor(
                    query_treatment_grid(env.do_value_distribution)
                )
                tasks = [
                    ActiveLearningTask(
                        0,
                        env,
                        InterventionalQuery(a.treatment, a.outcome, psi),
                        question_class(example.graph, a.treatment, a.outcome),
                    )
                ]
                example_meta = {
                    "example_file": example_path,
                    "example_model": example_model_to_dict(example),
                }
            else:
                if a.dense_value_diagnostic:
                    parser.error(
                        "--dense_value_diagnostic is for a fixed graph only"
                    )
                spec = ExperimentSpec(
                    name="prior",
                    task_source_kind="prior",
                    num_tasks=a.n_tasks or 50,
                    restarts_per_task=len(restarts),
                    initial_observational_rows=1,
                    strategies=names,
                    treatment=a.treatment,
                    outcome=a.outcome,
                )
                offset = a.task_offset or 0
                tasks = build_tasks(
                    spec,
                    data_cfg,
                    a.seed,
                    list(range(offset, offset + (a.n_tasks or 50))),
                )
            initial_rows = (
                a.n_obs if a.n_obs is not None else default_n_obs(data_cfg)
            )
        strategies = make_strategies(configs, names, value_searches)
    except ValueError as error:
        parser.error(str(error))
    if initial_rows < a.minimum_observational_rows:
        parser.error(
            f"the initial context has {initial_rows} observational rows, fewer "
            f"than the {a.minimum_observational_rows} the model's "
            "standardisation accepts; pass --minimum_observational_rows "
            f"{initial_rows} to run anyway (the standardisation is then "
            "estimated from fewer rows than in training)"
        )
    device = a.device or _default_device()
    D = tasks[0].environment.num_variables
    model = load_model(a.ckpt, a.model_config, device, num_nodes=D)
    try:
        adapter = CITNPPredictiveModel(
            model,
            device,
            fantasy_row_sampling=a.fantasy_row_sampling,
            inference_batch_size=strategies[0].config.inference_batch_size,
            minimum_observational_rows=a.minimum_observational_rows,
        )
    except ValueError as error:
        parser.error(str(error))
    support = training_support_from_data_config(data_cfg, model)
    print(f"training support: {support}")
    t0 = time.time()
    tables = run_experiment(
        adapter,
        setting,
        tasks,
        restarts,
        strategies,
        initial_rows,
        seed=a.seed,
        training_support=support,
        surrogate_steps=_ints(a.surrogate_steps) or [],
        dense_value_diagnostic_steps=(
            _ints(a.dense_value_diagnostic_steps) or []
            if a.dense_value_diagnostic
            else []
        ),
        dense_value_diagnostic_points=a.dense_value_diagnostic_points,
        progress=print,
    )
    for name, df in tables.as_dict().items():
        df.to_csv(f"{prefix}_{name}.csv", index=False)
    summary = summarise(tables.steps)
    summary.to_csv(f"{prefix}_summary.csv", index=False)

    def config_meta(path: str) -> dict:
        return {
            "path": path,
            "sha256": _sha256(path),
            "fingerprint": config_fingerprint(path),
        }

    meta = {
        "setting": setting,
        "experiment": None if a.experiment is None else config_meta(a.experiment),
        "model_config": config_meta(a.model_config),
        "data_config": config_meta(a.data_config),
        "ckpt": a.ckpt,
        "seed": a.seed,
        **(example_meta or {}),
        "tasks": [
            {
                "task": t.task,
                "treatment": t.query.treatment,
                "outcome": t.query.outcome,
                "relation": t.relation,
                "num_variables": t.environment.num_variables,
                "treatment_values": t.query.treatment_values.tolist(),
            }
            for t in tasks
        ],
        "restarts": restarts,
        "initial_observational_rows": initial_rows,
        "strategies": {s.label: asdict(s.config) for s in strategies},
        "training_support": asdict(support),
        "observational_prediction": adapter.supports_observational_prediction,
        "git_commit": _git_commit(),
        "args": vars(a),
        "seconds": time.time() - t0,
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"wrote {prefix}_*.csv and {prefix}_meta.json")
    if not summary.empty:
        print(
            summary[["strategy", "step", "num_runs", "mean_query_kl"]]
            .groupby("strategy")
            .tail(1)
            .to_string(index=False)
        )
    _plot(prefix)


def _plot(prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from causal_active_learning.evaluation.plots import plot_active_learning_all

    for path in plot_active_learning_all(prefix):
        print(f"wrote {path}")


def merge_command(a: argparse.Namespace) -> None:
    from causal_active_learning.evaluation.summaries import merge_runs

    summary = merge_runs(a.inputs, a.out)
    print(
        f"merged {len(a.inputs)} parts into {a.out}_*.csv "
        f"({int(summary.num_runs.max()) if len(summary) else 0} runs)"
    )
    _plot(a.out)


def plot_panels_command(a: argparse.Namespace, parser) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from causal_active_learning.evaluation.plots import plot_panels
    from causal_active_learning.integrations.citnp.experiment import (
        load_experiment,
    )

    if len(a.summaries) != len(a.experiments):
        parser.error("give one experiment file per summary")
    titles = a.titles or [None] * len(a.summaries)
    if len(titles) != len(a.summaries):
        parser.error("give one title per summary")
    specs = [load_experiment(path) for path in a.experiments]
    titles = [t or s.name for t, s in zip(titles, specs)]
    summaries = [pd.read_csv(path) for path in a.summaries]
    out = plot_panels(
        summaries,
        [s.graph_description() for s in specs],
        titles,
        a.out,
        all_bands=a.all_bands,
    )
    print(f"wrote {out}")


COMMANDS = {
    "run": (run_parser, "run one experiment (one setting), or an ad hoc run"),
    "merge": (merge_parser, "merge partial runs and recompute the summary"),
    "plot-panels": (plot_panels_parser, "ABCI Figure 4 layout from summaries"),
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
    elif a.command == "merge":
        merge_command(a)
    elif a.command == "plot-panels":
        plot_panels_command(a, parsers["plot-panels"])
    else:
        _plot(a.prefix)
