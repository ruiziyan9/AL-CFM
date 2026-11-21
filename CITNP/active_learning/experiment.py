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

from CITNP.active_learning.config import ActiveLearningConfig
from CITNP.active_learning.environment import LinearSCMEnvironment
from CITNP.active_learning.fantasy import ExperimentContext
from CITNP.active_learning.loop import (
    ActiveLearningGenerators,
    ActiveLearningStep,
    run_active_learning,
)
from CITNP.active_learning.predictive_model import PredictiveModel
from CITNP.active_learning.training_support import TrainingSupport

# Seed stream of the initial observational context (distinct from the
# loop's streams 1-6).
INITIAL_CONTEXT_STREAM = 11
# Seed stream of the (T, Y) pair drawn for a prior task.
QUERY_PAIR_STREAM = 12
# Points of the saved GP posterior per node and step.
SURROGATE_GRID_POINTS = 101
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


@dataclass
class ActiveLearningTables:
    """Results as tables (one CSV each).

    Attributes:
    ----------
    steps : pd.DataFrame
        One row per task x strategy x seed x step.

    scores : pd.DataFrame
        One row per task x strategy x seed x step x candidate node (EIG
        strategies only).

    trace : pd.DataFrame
        One row per BO evaluation (EIG strategies with BO only).

    surrogate : pd.DataFrame
        The final GP posterior of every node on a grid (BO only).

    dense : pd.DataFrame
        Dense value diagnostic (when requested).
    """

    steps: pd.DataFrame
    scores: pd.DataFrame
    trace: pd.DataFrame
    surrogate: pd.DataFrame
    dense: pd.DataFrame


def _step_rows(ident: dict, step: ActiveLearningStep, context_rows: int):
    c = step.chosen
    steps = {
        **ident, "step": step.step,
        "chosen_node": -1 if c.target_node is None else c.target_node,
        "chosen_value": np.nan if c.intervention_value is None
        else c.intervention_value,
        "chosen": c.describe(),
        "choice_is_clear": step.choice_is_clear,
        "predicted_information_gain": step.predicted_information_gain,
        "realised_information_gain": step.realised_information_gain,
        "kl_to_truth_before": step.kl_to_truth_before,
        "kl_to_truth_after": step.kl_to_truth_after,
        "context_rows": context_rows,
        "outside_training_support": step.outside_training_support,
        "support_violations": "; ".join(step.support_violations),
    }  # fmt: skip
    scores, trace, surrogate, dense = [], [], [], []
    at = {**ident, "step": step.step}
    if step.acquisition == "expected_information_gain":
        for s in step.scores:
            inner = s.inner_result
            scores.append({
                **at,
                "candidate_node": -1 if s.candidate.target_node is None
                else s.candidate.target_node,
                "optimised_value": np.nan
                if s.candidate.intervention_value is None
                else s.candidate.intervention_value,
                "optimised_standardised_value": np.nan if inner is None
                else inner.best_standardised_value,
                "expected_information_gain": s.expected_information_gain,
                "standard_error": s.standard_error,
                "coherence_gap": s.coherence_gap,
                "surrogate_estimate": np.nan if inner is None
                else inner.best_value_posterior_mean,
                "is_chosen": s.candidate == c,
            })  # fmt: skip
            if inner is None or inner.surrogate is None:
                continue
            for index, e in enumerate(inner.evaluations):
                trace.append({
                    **at, "node": inner.target_node, "evaluation": index,
                    "round": e.round, "standardised_value": e.standardised_value,
                    "value": e.value,
                    "expected_information_gain": e.expected_information_gain,
                    "standard_error": e.standard_error,
                    "acquisition_value": e.acquisition_value,
                    "lengthscale": e.lengthscale,
                    "output_scale": e.output_scale,
                })  # fmt: skip
            d = inner.domain
            grid = torch.linspace(
                d.lower, d.upper, SURROGATE_GRID_POINTS, dtype=torch.float64
            )
            mean, sd = inner.surrogate.posterior(grid[:, None])
            for u, m_, s_ in zip(grid.tolist(), mean.tolist(), sd.tolist()):
                surrogate.append({
                    **at, "node": inner.target_node, "standardised_value": u,
                    "value": d.to_value(u), "posterior_mean": m_,
                    "posterior_standard_deviation": s_,
                })  # fmt: skip
    for node, evaluations in (step.dense_value_diagnostic or {}).items():
        for e in evaluations:
            dense.append({
                **at, "node": node, "standardised_value": e.standardised_value,
                "value": e.value,
                "expected_information_gain": e.expected_information_gain,
                "standard_error": e.standard_error,
            })  # fmt: skip
    return steps, scores, trace, surrogate, dense


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
                    parts = _step_rows(ident, record, rows)
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
