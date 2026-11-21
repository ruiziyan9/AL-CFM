"""Evaluation of active learning strategies over tasks and seeds: the work
behind `python -m CITNP.evaluation active-learning` (plan section 8).

For every task (an environment and a query), seed and strategy: draw the
initial observational context (the same for every strategy), run the loop,
and collect the records into tables (`ActiveLearningTables`). Writing files,
printing and plotting are left to the caller.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentContext
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
from causal_active_learning.protocols import PredictiveModel
from causal_active_learning.support_check import TrainingSupport

# Seed stream of the initial observational context (distinct from the
# loop's streams 1-6).
INITIAL_CONTEXT_STREAM = 11
# Seed stream of the (T, Y) pair drawn for a prior task.
QUERY_PAIR_STREAM = 12
# Strategies whose inner level depends on the value optimiser.
OPTIMISED_ACQUISITIONS = ("expected_information_gain", "exhaustive_true_gain")


@dataclass
class ActiveLearningTask:
    """One environment and query.

    Attributes:
    ----------
    task : int
        Task index (part of every seed).

    environment : LinearSCMEnvironment
        The true SCM.

    treatment, outcome : int
        The query X_outcome | do(X_treatment = psi).

    relation : str
        Question class of (treatment, outcome) in the true graph, or "".
    """

    task: int
    environment: LinearSCMEnvironment
    treatment: int
    outcome: int
    relation: str = ""


@dataclass
class ActiveLearningStrategy:
    """A labelled loop configuration (label e.g.
    "expected_information_gain/bayesian_optimisation" or "random")."""

    label: str
    config: ActiveLearningConfig


def make_strategies(
    base_config: ActiveLearningConfig,
    acquisitions: Sequence[str],
    value_optimisers: Sequence[str],
) -> list[ActiveLearningStrategy]:
    """One strategy per acquisition, and per value optimiser for the
    acquisitions with an inner level (OPTIMISED_ACQUISITIONS)."""
    out = []
    for acquisition in acquisitions:
        if acquisition in OPTIMISED_ACQUISITIONS:
            for optimiser in value_optimisers:
                out.append(
                    ActiveLearningStrategy(
                        f"{acquisition}/{optimiser}",
                        replace(
                            base_config,
                            acquisition=acquisition,
                            value_optimiser=optimiser,
                        ),
                    )  # fmt: skip
                )
        else:
            out.append(
                ActiveLearningStrategy(
                    acquisition, replace(base_config, acquisition=acquisition)
                )
            )
    return out


def query_treatment_grid(
    do_value_distribution: dict, num_values: int = 13, grid_eps: float = 0.02
) -> np.ndarray:
    """The psi grid of the k-shot evaluations: quantiles
    linspace(grid_eps, 1 - grid_eps, num_values) of the task's do-value
    distribution (`do_dist_quantile`)."""
    from CITNP.datasets.mixed_regime_generator import do_dist_quantile

    levels = np.linspace(grid_eps, 1.0 - grid_eps, num_values)
    return np.asarray(do_dist_quantile(do_value_distribution, levels), float)


def prior_tasks(
    data_config,
    num_variables: int,
    task_indices: Sequence[int],
    base_seed: int,
    treatment: int | None = None,
    outcome: int | None = None,
    fixed_graph: np.ndarray | None = None,
) -> list[ActiveLearningTask]:
    """Tasks of the training prior (the `kshot-prior` sampler, `prior_scm`).
    With treatment and outcome None, each task's (T, Y) is an ordered pair
    of distinct nodes drawn uniformly from rng [base_seed, D, i,
    QUERY_PAIR_STREAM] (kshot-prior scores every pair; the loop needs one)."""
    from CITNP.prior_diagnostics.stats import question_class

    tasks = []
    for i in task_indices:
        environment = LinearSCMEnvironment.from_prior_task(
            data_config, i, base_seed, num_variables, fixed_graph=fixed_graph
        )
        T, Y = treatment, outcome
        if T is None or Y is None:
            rng = np.random.default_rng(
                [base_seed, num_variables, i, QUERY_PAIR_STREAM]
            )
            T, Y = (int(v) for v in rng.choice(num_variables, 2, replace=False))
        graph = environment.graph
        relation = "" if graph is None else question_class(graph, T, Y)
        tasks.append(ActiveLearningTask(i, environment, T, Y, relation))
    return tasks


def run_active_learning_experiment(
    predictive_model: PredictiveModel,
    tasks: Sequence[ActiveLearningTask],
    strategies: Sequence[ActiveLearningStrategy],
    seeds: Sequence[int],
    num_observational_rows: int,
    *,
    base_seed: int = 9101,
    training_support: TrainingSupport | None = None,
    query_grid_points: int = 13,
    grid_eps: float = 0.02,
    dense_value_diagnostic_steps: Sequence[int] = (),
    dense_value_diagnostic_points: int = 41,
    progress: Callable[[str], None] | None = None,
) -> ActiveLearningTables:
    """Run every strategy on every task and seed.

    Per (task, seed): the initial context is `num_observational_rows`
    observational rows from rng [base_seed, task, seed,
    INITIAL_CONTEXT_STREAM], and the loop's generators are
    `ActiveLearningGenerators.from_seed([base_seed, task, seed])`, the same
    for every strategy (so all strategies start from the same context and
    environment random state). The psi grid is `query_treatment_grid` of the
    task's do-value distribution.
    """
    steps, scores, trace, surrogate, dense = [], [], [], [], []
    for task in tasks:
        env = task.environment
        if env.do_value_distribution is None:
            raise ValueError(
                "the task's environment has no do-value distribution"
            )
        psi = torch.as_tensor(
            query_treatment_grid(
                env.do_value_distribution, query_grid_points, grid_eps
            ),
            dtype=torch.float64,
        )
        for seed in seeds:
            rng = np.random.default_rng(
                [base_seed, task.task, seed, INITIAL_CONTEXT_STREAM]
            )
            values = env.sample_observational(num_observational_rows, rng)
            initial = ExperimentContext(
                values, torch.zeros(values.shape, dtype=torch.bool)
            )
            for strategy in strategies:
                if progress is not None:
                    progress(
                        f"task {task.task} (X{task.outcome + 1} | "
                        f"do(X{task.treatment + 1})), seed {seed}, "
                        f"{strategy.label}"
                    )
                cfg = strategy.config
                generators = ActiveLearningGenerators.from_seed(
                    [base_seed, task.task, seed]
                )
                records = run_active_learning(
                    predictive_model, env, initial, task.treatment,
                    task.outcome, psi, cfg, generators,
                    training_support=training_support,
                    prior_value_range=env.prior_value_range(),
                    dense_value_diagnostic_steps=dense_value_diagnostic_steps,
                    dense_value_diagnostic_points=dense_value_diagnostic_points,
                    progress=progress,
                )  # fmt: skip
                ident = {
                    "task": task.task, "T": task.treatment, "Y": task.outcome,
                    "relation": task.relation, "strategy": strategy.label,
                    "acquisition": cfg.acquisition,
                    "value_optimiser": cfg.value_optimiser
                    if cfg.acquisition in OPTIMISED_ACQUISITIONS else "",
                    "seed": seed,
                }  # fmt: skip
                rows = initial.num_rows
                for record in records:
                    rows += record.real_rows.shape[0]
                    parts = step_rows(ident, record, rows)
                    steps.append(parts[0])
                    scores += parts[1]
                    trace += parts[2]
                    surrogate += parts[3]
                    dense += parts[4]
    return ActiveLearningTables(
        steps=pd.DataFrame(steps),
        scores=pd.DataFrame(scores),
        trace=pd.DataFrame(trace),
        surrogate=pd.DataFrame(surrogate),
        dense=pd.DataFrame(dense),
    )
