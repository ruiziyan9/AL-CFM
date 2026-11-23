"""The active learning loop.

Step 0 records the query prediction and its KL to the truth on the initial
context; each later step:
1. predicts the query p(Y | do(X_T = psi), D_t) on the psi values;
2. chooses the acquisition (`strategies.choose_acquisition`);
3. runs it on the (constrained) environment and appends all R real rows;
4. predicts the query on D_{t+1}; records the realised information gain
   KL(after || before), the query KL to the truth, the support check and the
   size of the newest regime.

Randomness: `ActiveLearningGenerators` holds separate streams, so changing the
acquisition settings does not change the real rows drawn for a given strategy
and seed.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from causal_active_learning.acquisitions import (
    Acquisition,
    ConstrainedEnvironment,
    InterventionalQuery,
    resolve_intervenable_nodes,
)
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import (
    UnivariateGaussianMixture,
    kl_divergence_quadrature,
)
from causal_active_learning.information_gain import predict_query_on_context
from causal_active_learning.protocols import (
    CausalPredictiveModel,
    ExperimentEnvironment,
)
from causal_active_learning.selection import (
    OBSERVATIONAL_UNAVAILABLE_WARNING,
    AcquisitionScore,
)
from causal_active_learning.strategies import choose_acquisition, query_kl
from causal_active_learning.support_check import (
    TrainingSupport,
    check_num_variables,
    context_violations,
    worst_case_violations,
)
from causal_active_learning.value_search import ValueEvaluation

# Stream numbers of `ActiveLearningGenerators.from_seed`.
_STREAMS = {
    "inner_fantasy": 1,
    "outer_fantasy": 2,
    "bayesian_optimisation": 3,
    "environment": 4,
    "baseline": 5,
    "diagnostic": 6,
}


def _torch_generator(key: Sequence[int]) -> torch.Generator:
    state = np.random.SeedSequence(list(key)).generate_state(2)
    seed = (int(state[0]) << 32 | int(state[1])) & ((1 << 63) - 1)
    return torch.Generator().manual_seed(seed)


@dataclass
class ActiveLearningGenerators:
    """Separate random streams of one loop run.

    Attributes:
    ----------
    inner_fantasy : torch.Generator
        Fantasy noise of the inner level (one draw per node per step).

    outer_fantasy : torch.Generator
        Fresh fantasy noise of the outer re-evaluation.

    bayesian_optimisation : torch.Generator
        GP hyperparameter restarts.

    environment : np.random.Generator
        The true SCM's noise (`LinearSCM.sample` takes a numpy generator).

    baseline : torch.Generator
        Random choices of the "random" and "treatment" strategies.

    diagnostic : torch.Generator
        Fantasy noise of the dense value diagnostic.
    """

    inner_fantasy: torch.Generator
    outer_fantasy: torch.Generator
    bayesian_optimisation: torch.Generator
    environment: np.random.Generator
    baseline: torch.Generator
    diagnostic: torch.Generator

    @classmethod
    def from_seed(cls, seed: int | Sequence[int]) -> "ActiveLearningGenerators":
        """Independent streams keyed by (seed..., stream number)."""
        key = [int(seed)] if isinstance(seed, int | np.integer) else list(seed)
        return cls(
            inner_fantasy=_torch_generator([*key, _STREAMS["inner_fantasy"]]),
            outer_fantasy=_torch_generator([*key, _STREAMS["outer_fantasy"]]),
            bayesian_optimisation=_torch_generator(
                [*key, _STREAMS["bayesian_optimisation"]]
            ),
            environment=np.random.default_rng([*key, _STREAMS["environment"]]),
            baseline=_torch_generator([*key, _STREAMS["baseline"]]),
            diagnostic=_torch_generator([*key, _STREAMS["diagnostic"]]),
        )


@dataclass
class ActiveLearningStep:
    """The record of one step.

    Attributes:
    ----------
    step : int
        0 = the initial context; after step t the context holds t acquisitions.

    strategy : str
        The strategy.

    num_variables : int
        D.

    intervenable_nodes : tuple[int, ...]
        The intervenable nodes of the run.

    scores : list[AcquisitionScore]
        Candidate scores (EIG strategies; empty for baselines and step 0).

    chosen : Acquisition | None
        The acquisition run (None at step 0).

    choice_is_clear : bool | None
        See `choice_is_clear` (None for baselines and step 0).

    observational_candidate_available : bool
        Whether the strategy could choose the observational candidate (for the
        EIG strategy: only if the model can predict observational rows).

    real_rows : torch.Tensor | None
        [R, D] rows from the environment (None at step 0).

    query_after : UnivariateGaussianMixture
        The query [Q, K] on the context after this step.

    realised_information_gain : float | None
        psi-averaged KL(after || before) (None at step 0).

    query_kl_to_truth : float | None
        psi-averaged KL(truth || model) after this step (None without truth).

    outside_training_support : bool
        The context after this step lies outside the training support.

    newest_regime_size : int | None
        Rows intervened on the node of the newest acquisition (None if
        observational).

    predicted_information_gain : float
        Re-estimated EIG of the chosen acquisition (NaN otherwise).

    support_violations : list[str]
        How the context leaves the support.

    dense_value_diagnostic : dict[int, list[ValueEvaluation]] | None
        Exhaustive EIG per node on a dense value grid, when requested.
    """

    step: int
    strategy: str
    num_variables: int
    intervenable_nodes: tuple[int, ...]
    scores: list[AcquisitionScore]
    chosen: Acquisition | None
    choice_is_clear: bool | None
    observational_candidate_available: bool
    real_rows: torch.Tensor | None
    query_after: UnivariateGaussianMixture
    realised_information_gain: float | None
    query_kl_to_truth: float | None
    outside_training_support: bool
    newest_regime_size: int | None
    predicted_information_gain: float = float("nan")
    support_violations: list[str] | None = None
    dense_value_diagnostic: dict[int, list[ValueEvaluation]] | None = None


def _num_target_nodes(strategy: str, nodes: Sequence[int]) -> int:
    return {"observational": 0, "treatment": 1}.get(strategy, len(nodes))


def run_active_learning(
    model: CausalPredictiveModel,
    environment: ExperimentEnvironment,
    initial_context: ExperimentalContext,
    query: InterventionalQuery,
    config: ActiveLearningConfig,
    generators: ActiveLearningGenerators,
    *,
    training_support: TrainingSupport | None = None,
    dense_value_diagnostic_steps: Sequence[int] = (),
    dense_value_diagnostic_points: int = 41,
    progress: Callable[[str], None] | None = None,
) -> list[ActiveLearningStep]:
    """Run step 0 and config.acquisition_budget acquisitions of strategy
    config.strategy. Returns budget + 1 records.

    The environment is always wrapped in `ConstrainedEnvironment` with the
    intervenable nodes. Every strategy uses the same environment generator,
    so strategies run with the same seed start from the same state.

    Args:
    ----------
    model, environment
        The model and the true system.

    initial_context : ExperimentalContext
        D_0.

    query : InterventionalQuery
        The target query and its psi values.

    config : ActiveLearningConfig
        Settings.

    generators : ActiveLearningGenerators
        Random streams.

    training_support : TrainingSupport | None
        If given: D above the architecture's maximum raises before the loop;
        contexts outside the support give a warning and the step's flag; its
        do-value range is the "training_range" value domain.

    dense_value_diagnostic_steps, dense_value_diagnostic_points
        Steps (1-based) at which the EIG strategy also runs the dense value
        diagnostic with this many values per node.

    progress : Callable[[str], None] | None
        Called with one line per step (e.g. print).
    """
    D = initial_context.num_variables
    nodes = resolve_intervenable_nodes(D, config.intervenable_nodes)
    training_range = None
    if training_support is not None:
        check_num_variables(D, training_support)  # raises above the maximum
        training_range = training_support.do_value_range
        problems = worst_case_violations(
            initial_context,
            config.acquisition_budget,
            config.rows_per_acquisition,
            _num_target_nodes(config.strategy, nodes),
            training_support,
        )
        for problem in problems:
            warnings.warn(
                f"active learning may leave the training support: {problem}",
                stacklevel=2,
            )
    environment = ConstrainedEnvironment(environment, nodes)
    if config.strategy == "expected_information_gain":
        available = config.include_observational_candidate and bool(
            getattr(model, "supports_observational_prediction", False)
        )
        if config.include_observational_candidate and not available:
            warnings.warn(OBSERVATIONAL_UNAVAILABLE_WARNING, stacklevel=2)
    else:
        available = config.include_observational_candidate
    psi = query.treatment_values
    truth = environment.true_query_distribution(
        query.treatment, query.outcome, psi
    )
    points = config.quadrature_points

    def check(context, step):
        if training_support is None:
            return []
        problems = context_violations(context, training_support)
        for problem in problems:
            warnings.warn(
                f"step {step}: outside the training support: {problem}",
                stacklevel=3,
            )
        return problems

    context = initial_context
    before = predict_query_on_context(
        model, context, query.treatment, query.outcome, psi
    )
    violations = check(context, 0)
    steps = [
        ActiveLearningStep(
            step=0,
            strategy=config.strategy,
            num_variables=D,
            intervenable_nodes=nodes,
            scores=[],
            chosen=None,
            choice_is_clear=None,
            observational_candidate_available=available,
            real_rows=None,
            query_after=before,
            realised_information_gain=None,
            query_kl_to_truth=None
            if truth is None
            else query_kl(truth, before, points),
            outside_training_support=bool(violations),
            newest_regime_size=context.newest_regime_size(),
            support_violations=violations,
        )
    ]
    for step in range(1, config.acquisition_budget + 1):
        choice = choose_acquisition(
            model,
            environment,
            context,
            query,
            config,
            generators,
            query_before=before,
            truth=truth,
            training_value_range=training_range,
            dense_value_diagnostic_points=(
                dense_value_diagnostic_points
                if step in dense_value_diagnostic_steps
                else None
            ),
        )
        chosen = choice.chosen
        rows = environment.run(chosen, generators.environment)
        context = context.append(rows, chosen.target_node)
        after = predict_query_on_context(
            model, context, query.treatment, query.outcome, psi
        )
        violations = check(context, step)
        record = ActiveLearningStep(
            step=step,
            strategy=config.strategy,
            num_variables=D,
            intervenable_nodes=nodes,
            scores=choice.scores,
            chosen=chosen,
            choice_is_clear=choice.choice_is_clear,
            observational_candidate_available=available,
            real_rows=rows,
            query_after=after,
            realised_information_gain=float(
                kl_divergence_quadrature(after, before, points).mean()
            ),
            query_kl_to_truth=None
            if truth is None
            else query_kl(truth, after, points),
            outside_training_support=bool(violations),
            newest_regime_size=context.newest_regime_size()
            if chosen.target_node is not None
            else None,
            predicted_information_gain=choice.predicted_information_gain,
            support_violations=violations,
            dense_value_diagnostic=choice.dense_value_diagnostic,
        )
        steps.append(record)
        if progress is not None:
            predicted = record.predicted_information_gain
            kl = record.query_kl_to_truth
            progress(
                f"[{config.strategy}] step {step}: {chosen.describe()}"
                + (
                    f", predicted {config.information_gain_objective} "
                    f"{predicted:.4f}"
                    + ("" if record.choice_is_clear else " (not clear-cut)")
                    if math.isfinite(predicted)
                    else ""
                )
                + f", realised gain {record.realised_information_gain:.4f}"
                + ("" if kl is None else f", query KL {kl:.4f}")
            )
        before = after
    return steps
