"""Experiment configs (YAML) and the runner of the CITNP integration.

An experiment YAML holds everything one setting (one panel) needs:

    name: str                         # setting name, written to every row
    task_source:
      kind: fixed_graph | prior
      num_variables: int              # fixed_graph
      edges: list of [parent, child]  # fixed_graph, 0-based
      num_tasks: int                  # ground-truth SCMs
      restarts_per_task: int
    query:
      treatment: int                  # 0-based; optional for prior tasks
      outcome: int
      treatment_values: list[float]   # or the two keys below
      treatment_range: [low, high]
      num_query_samples: int
    initial_observational_rows: int
    strategies: list[str]
    active_learning: { ActiveLearningConfig fields }

It is read with CITNP's `load_config_mapping`, so `extends` works; top-level
keys of a child replace the parent's whole (nested sections are not merged).

Tasks, restarts and seeds. Task t is one ground-truth SCM: the fixed graph
with mechanism parameters drawn from the data config's prior for (seed, t)
(`prior_scm`, the kshot machinery), or task t of the prior. Its psi values
come from rng (seed, t). Restart r of task t keeps the SCM and redraws the
initial observational rows (rng (seed, t, r, 11)) and the environment stream
(rng (seed, t, r, 13)); the acquisition randomness comes from
(seed, t, r, crc32(strategy)). So every strategy and every setting run with
the same seed sees the same SCMs, psi values and initial rows per (t, r), and
the environment stream starts from the same state: comparisons are paired.
"""

import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace

import numpy as np
import torch

from causal_active_learning.acquisitions import InterventionalQuery
from causal_active_learning.config import STRATEGIES, ActiveLearningConfig
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.evaluation.records import (
    ActiveLearningTables,
    step_rows,
)
from causal_active_learning.integrations.citnp.environment import (
    LinearSCMEnvironment,
)
from causal_active_learning.loop import (
    ActiveLearningGenerators,
    run_active_learning,
)
from causal_active_learning.protocols import CausalPredictiveModel
from causal_active_learning.support_check import TrainingSupport

# Seed streams.
INITIAL_CONTEXT_STREAM = 11
QUERY_PAIR_STREAM = 12
ENVIRONMENT_STREAM = 13
QUERY_VALUES_STREAM = 14
NUM_VARIABLES_STREAM = 15
# Strategies whose inner search depends on the value search.
OPTIMISED_STRATEGIES = ("expected_information_gain", "exhaustive_true_gain")

_TOP_KEYS = {
    "name",
    "task_source",
    "query",
    "initial_observational_rows",
    "strategies",
    "active_learning",
}
_TASK_SOURCE_KEYS = {
    "kind",
    "num_variables",
    "edges",
    "num_tasks",
    "restarts_per_task",
}
_QUERY_KEYS = {
    "treatment",
    "outcome",
    "treatment_values",
    "treatment_range",
    "num_query_samples",
}


@dataclass
class ExperimentSpec:
    """A validated experiment config (see the module docstring)."""

    name: str
    task_source_kind: str
    num_tasks: int
    restarts_per_task: int
    initial_observational_rows: int
    strategies: list[str]
    active_learning: dict = field(default_factory=dict)
    num_variables: int | None = None
    edges: list[tuple[int, int]] | None = None
    treatment: int | None = None
    outcome: int | None = None
    treatment_values: list[float] | None = None
    treatment_range: tuple[float, float] | None = None
    num_query_samples: int = 16

    def config(self, strategy: str, **overrides) -> ActiveLearningConfig:
        """The ActiveLearningConfig of one strategy."""
        settings = {**self.active_learning, **overrides, "strategy": strategy}
        return ActiveLearningConfig(**settings)

    def adjacency(self) -> np.ndarray | None:
        """[D, D] 0/1 adjacency of the fixed graph (None for prior tasks)."""
        if self.edges is None or self.num_variables is None:
            return None
        adjacency = np.zeros((self.num_variables, self.num_variables), int)
        for parent, child in self.edges:
            adjacency[parent, child] = 1
        return adjacency

    def graph_description(self) -> dict:
        """What `plot_panels` needs to draw the graph."""
        nodes = self.active_learning.get("intervenable_nodes")
        return {
            "num_variables": self.num_variables,
            "edges": self.edges or [],
            "treatment": self.treatment,
            "outcome": self.outcome,
            "intervenable_nodes": None if nodes is None else list(nodes),
        }


def _check_keys(section: dict, allowed: set, where: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown}; allowed {sorted(allowed)}"
        )


def _is_acyclic(num_variables: int, edges) -> bool:
    children: dict[int, list[int]] = {n: [] for n in range(num_variables)}
    indegree = dict.fromkeys(range(num_variables), 0)
    for parent, child in edges:
        children[parent].append(child)
        indegree[child] += 1
    stack = [n for n, d in indegree.items() if d == 0]
    seen = 0
    while stack:
        node = stack.pop()
        seen += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(child)
    return seen == num_variables


def experiment_from_mapping(
    mapping: dict, where: str = "experiment"
) -> ExperimentSpec:
    """Validate an experiment mapping (unknown keys raise; exactly one of
    treatment_values and treatment_range; edges acyclic and in range; the
    active_learning section is a valid ActiveLearningConfig)."""
    _check_keys(mapping, _TOP_KEYS, where)
    for key in (
        "name",
        "task_source",
        "query",
        "initial_observational_rows",
        "strategies",
    ):
        if key not in mapping:
            raise ValueError(f"{where}: missing key {key!r}")
    source = dict(mapping["task_source"])
    _check_keys(source, _TASK_SOURCE_KEYS, f"{where}: task_source")
    query = dict(mapping["query"])
    _check_keys(query, _QUERY_KEYS, f"{where}: query")
    kind = source.get("kind")
    if kind not in ("fixed_graph", "prior"):
        raise ValueError(
            f"{where}: task_source.kind must be fixed_graph or prior"
        )
    spec = ExperimentSpec(
        name=str(mapping["name"]),
        task_source_kind=kind,
        num_tasks=int(source.get("num_tasks", 1)),
        restarts_per_task=int(source.get("restarts_per_task", 1)),
        initial_observational_rows=int(mapping["initial_observational_rows"]),
        strategies=[str(s) for s in mapping["strategies"]],
        active_learning=dict(mapping.get("active_learning") or {}),
        treatment=None
        if query.get("treatment") is None
        else int(query["treatment"]),
        outcome=None if query.get("outcome") is None else int(query["outcome"]),
        num_query_samples=int(query.get("num_query_samples", 16)),
    )
    if spec.num_tasks < 1 or spec.restarts_per_task < 1:
        raise ValueError(f"{where}: num_tasks and restarts_per_task must be >= 1")
    if spec.initial_observational_rows < 1:
        raise ValueError(f"{where}: initial_observational_rows must be >= 1")
    unknown = [s for s in spec.strategies if s not in STRATEGIES]
    if unknown or not spec.strategies:
        raise ValueError(
            f"{where}: unknown strategies {unknown}; valid {STRATEGIES}"
        )
    has_values = query.get("treatment_values") is not None
    has_range = query.get("treatment_range") is not None
    if has_values == has_range:
        raise ValueError(
            f"{where}: give exactly one of query.treatment_values and "
            "query.treatment_range"
        )
    if has_values:
        spec.treatment_values = [float(v) for v in query["treatment_values"]]
    else:
        low, high = (float(v) for v in query["treatment_range"])
        if not low < high:
            raise ValueError(f"{where}: query.treatment_range needs low < high")
        spec.treatment_range = (low, high)
    if kind == "fixed_graph":
        if "num_variables" not in source or "edges" not in source:
            raise ValueError(
                f"{where}: fixed_graph needs num_variables and edges"
            )
        D = int(source["num_variables"])
        edges = [(int(p), int(c)) for p, c in source["edges"]]
        if any(not (0 <= p < D and 0 <= c < D) or p == c for p, c in edges):
            raise ValueError(
                f"{where}: edges must join distinct nodes in 0..{D - 1}"
            )
        if not _is_acyclic(D, edges):
            raise ValueError(f"{where}: the edges form a cycle")
        if spec.treatment is None or spec.outcome is None:
            raise ValueError(
                f"{where}: fixed_graph needs query.treatment and outcome"
            )
        if not (0 <= spec.treatment < D and 0 <= spec.outcome < D):
            raise ValueError(f"{where}: treatment and outcome must be below {D}")
        spec.num_variables, spec.edges = D, edges
    elif (spec.treatment is None) != (spec.outcome is None):
        raise ValueError(
            f"{where}: give both query.treatment and outcome, or neither"
        )
    if spec.treatment is not None and spec.treatment == spec.outcome:
        raise ValueError(f"{where}: treatment and outcome must differ")
    allowed = {f.name for f in fields(ActiveLearningConfig)} - {"strategy"}
    _check_keys(spec.active_learning, allowed, f"{where}: active_learning")
    for strategy in spec.strategies:  # validates every config
        spec.config(strategy)
    return spec


def load_experiment(path: str) -> ExperimentSpec:
    """Read and validate an experiment YAML (with `extends`)."""
    from CITNP.utils.config_files import load_config_mapping

    return experiment_from_mapping(load_config_mapping(path), str(path))


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@dataclass
class ActiveLearningTask:
    """One ground-truth SCM with its query (psi values fixed per task)."""

    task: int
    environment: LinearSCMEnvironment
    query: InterventionalQuery
    relation: str = ""


def query_treatment_grid(
    do_value_distribution: dict, num_values: int = 13, grid_eps: float = 0.02
) -> np.ndarray:
    """The psi grid of the k-shot evaluations: quantiles
    linspace(grid_eps, 1 - grid_eps, num_values) of the task's do-value
    distribution (`do_dist_quantile`)."""
    from CITNP.datasets.mixed_regime_generator import do_dist_quantile

    levels = np.linspace(grid_eps, 1.0 - grid_eps, num_values)
    return np.asarray(do_dist_quantile(do_value_distribution, levels), float)


def _query_values(
    spec: ExperimentSpec, seed: int, task: int, env
) -> torch.Tensor:
    if spec.treatment_values is not None:
        values = np.asarray(spec.treatment_values, float)
    elif spec.treatment_range is not None:
        rng = np.random.default_rng([seed, task, QUERY_VALUES_STREAM])
        values = np.sort(
            rng.uniform(*spec.treatment_range, spec.num_query_samples)
        )
    else:
        values = query_treatment_grid(env.do_value_distribution)
    return torch.as_tensor(values, dtype=torch.float64)


def build_tasks(
    spec: ExperimentSpec, data_config, seed: int, task_indices: Sequence[int]
) -> list[ActiveLearningTask]:
    """The tasks `task_indices` of an experiment (see the module docstring).
    Prior tasks: D drawn from the data config's num_variables with rng
    (seed, t), (T, Y) from the query or a uniformly random ordered pair from
    rng (seed, D, t, 12); the question class is recorded."""
    from CITNP.prior_diagnostics.stats import question_class

    tasks = []
    for t in task_indices:
        if spec.task_source_kind == "fixed_graph":
            D = int(spec.num_variables)  # type: ignore[arg-type]
            env = LinearSCMEnvironment.from_prior_task(
                data_config, t, seed, D, fixed_graph=spec.adjacency()
            )
            assert spec.treatment is not None and spec.outcome is not None
            T, Y = spec.treatment, spec.outcome
        else:
            choices = [int(d) for d in data_config.num_variables]
            rng = np.random.default_rng([seed, t, NUM_VARIABLES_STREAM])
            D = int(rng.choice(choices))
            env = LinearSCMEnvironment.from_prior_task(data_config, t, seed, D)
            if spec.treatment is None or spec.outcome is None:
                pair = np.random.default_rng([seed, D, t, QUERY_PAIR_STREAM])
                T, Y = (int(v) for v in pair.choice(D, 2, replace=False))
            else:
                T, Y = spec.treatment, spec.outcome
        query = InterventionalQuery(T, Y, _query_values(spec, seed, t, env))
        relation = "" if env.graph is None else question_class(env.graph, T, Y)
        tasks.append(ActiveLearningTask(t, env, query, relation))
    return tasks


# --------------------------------------------------------------------------- #
# Strategies and the runner
# --------------------------------------------------------------------------- #
@dataclass
class ActiveLearningStrategy:
    """A labelled loop configuration."""

    label: str
    config: ActiveLearningConfig


def make_strategies(
    configs: Callable[[str], ActiveLearningConfig],
    strategies: Sequence[str],
    value_searches: Sequence[str] | None = None,
) -> list[ActiveLearningStrategy]:
    """One strategy per name; with several `value_searches`, the strategies
    with an inner search run once per value search, labelled
    "<strategy>/<value search>"."""
    out = []
    for name in strategies:
        config = configs(name)
        if (
            name in OPTIMISED_STRATEGIES
            and value_searches
            and len(value_searches) > 1
        ):
            for search in value_searches:
                out.append(
                    ActiveLearningStrategy(
                        f"{name}/{search}", replace(config, value_search=search)
                    )
                )
        else:
            if value_searches and len(value_searches) == 1:
                config = replace(config, value_search=value_searches[0])
            out.append(ActiveLearningStrategy(name, config))
    return out


def strategy_key(label: str) -> int:
    """A stable integer of a strategy label (seed of its acquisition
    randomness)."""
    return zlib.crc32(label.encode())


def run_experiment(
    model: CausalPredictiveModel,
    setting: str,
    tasks: Sequence[ActiveLearningTask],
    restart_indices: Sequence[int],
    strategies: Sequence[ActiveLearningStrategy],
    initial_observational_rows: int,
    *,
    seed: int = 0,
    training_support: TrainingSupport | None = None,
    surrogate_steps: Sequence[int] = (1,),
    dense_value_diagnostic_steps: Sequence[int] = (),
    dense_value_diagnostic_points: int = 41,
    progress: Callable[[str], None] | None = None,
) -> ActiveLearningTables:
    """Run every strategy on every task and restart (seeds as in the module
    docstring) and collect the step records into tables."""
    import pandas as pd

    rows: dict[str, list[dict]] = {}
    for task in tasks:
        env = task.environment
        for restart in restart_indices:
            rng = np.random.default_rng(
                [seed, task.task, restart, INITIAL_CONTEXT_STREAM]
            )
            values = env.sample_observational(initial_observational_rows, rng)
            initial = ExperimentalContext(
                values, torch.zeros(values.shape, dtype=torch.bool)
            )
            for strategy in strategies:
                if progress is not None:
                    progress(
                        f"{setting}: task {task.task} (X{task.query.outcome + 1}"
                        f" | do(X{task.query.treatment + 1})), restart "
                        f"{restart}, {strategy.label}"
                    )
                cfg = strategy.config
                generators = ActiveLearningGenerators.from_seed(
                    [seed, task.task, restart, strategy_key(strategy.label)]
                )
                generators.environment = np.random.default_rng(
                    [seed, task.task, restart, ENVIRONMENT_STREAM]
                )
                records = run_active_learning(
                    model,
                    env,
                    initial,
                    task.query,
                    cfg,
                    generators,
                    training_support=training_support,
                    dense_value_diagnostic_steps=dense_value_diagnostic_steps,
                    dense_value_diagnostic_points=dense_value_diagnostic_points,
                    progress=progress,
                )
                ident = {
                    "setting": setting,
                    "task": task.task,
                    "restart": restart,
                    "strategy": strategy.label,
                    "T": task.query.treatment,
                    "Y": task.query.outcome,
                    "relation": task.relation,
                    "value_search": cfg.value_search
                    if cfg.strategy in OPTIMISED_STRATEGIES
                    else "",
                }
                context_rows = initial.num_rows
                for record in records:
                    if record.real_rows is not None:
                        context_rows += record.real_rows.shape[0]
                    parts = step_rows(
                        ident,
                        record,
                        context_rows,
                        cfg.rows_per_acquisition,
                        surrogate_steps,
                        cfg.acquisition_grid_points,
                    )
                    for name, part in parts.items():
                        rows.setdefault(name, []).extend(part)
    return ActiveLearningTables(
        **{
            name: pd.DataFrame(rows.get(name, []))
            for name in (
                "steps",
                "scores",
                "bayesian_optimisation_trace",
                "surrogate",
                "dense_value_diagnostic",
            )
        }
    )
