"""The active learning loop.

One step, with D_t the real context:
1. predict the query p(Y | do(X_T = psi), D_t) on the psi values;
2. choose the acquisition (`strategies.choose_acquisition`; for the EIG
   strategy the bi-level selection of `selection.py`);
3. run it on the environment and append the real rows;
4. predict the query on D_{t+1}; record the realised information gain
   KL(after || before) and the query KL to the truth, both averaged over psi.

Randomness: `ActiveLearningGenerators` holds separate streams, so changing the
acquisition settings does not change the real data drawn for a given seed and
sequence of acquisitions.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from causal_active_learning.acquisitions import (
    Acquisition,
    candidate_target_nodes,
)
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import (
    UnivariateGaussianMixture,
    kl_divergence_quadrature,
)
from causal_active_learning.information_gain import predict_query_on_context
from causal_active_learning.protocols import (
    ExperimentEnvironment,
    CausalPredictiveModel,
)
from causal_active_learning.selection import AcquisitionScore
from causal_active_learning.strategies import (
    choose_acquisition,
    num_possible_target_nodes,
    query_kl,
)
from causal_active_learning.support_check import (
    TrainingSupport,
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
    """The record of one loop step.

    Attributes:
    ----------
    step : int
        1-based; after this step the context holds `step` experiments.

    acquisition : str
        The strategy.

    scores : list[AcquisitionScore]
        Outer-level scores (EIG strategy; the inner-level surrogate values
        for "exhaustive_true_gain"; empty for the other baselines).

    chosen : Acquisition
        The experiment run.

    choice_is_clear : bool
        See `choice_is_clear` (True for the baselines).

    predicted_information_gain : float
        Re-evaluated EIG of the chosen experiment (NaN for the baselines).

    real_rows : torch.Tensor
        [R, D] rows drawn from the environment.

    query_before, query_after : UnivariateGaussianMixture
        The query [V, K] before and after the rows were added.

    realised_information_gain : float
        psi-averaged KL(after || before).

    kl_to_truth_before, kl_to_truth_after : float
        psi-averaged KL(truth || model) before and after.

    outside_training_support : bool
        The context after the step lies outside the training prior's
        context shapes (`support_violations` says how).

    support_violations : list[str]
        See `context_violations`.

    dense_value_diagnostic : dict[int, list[ValueEvaluation]] | None
        Exhaustive EIG per node on a dense value grid, when requested.
    """

    step: int
    strategy: str
    scores: list[AcquisitionScore]
    chosen: Acquisition
    choice_is_clear: bool
    predicted_information_gain: float
    real_rows: torch.Tensor
    query_before: UnivariateGaussianMixture
    query_after: UnivariateGaussianMixture
    realised_information_gain: float
    kl_to_truth_before: float
    kl_to_truth_after: float
    outside_training_support: bool
    support_violations: list[str]
    dense_value_diagnostic: dict[int, list[ValueEvaluation]] | None = None


def run_active_learning(
    model: CausalPredictiveModel,
    environment: ExperimentEnvironment,
    initial_context: ExperimentalContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    generators: ActiveLearningGenerators,
    *,
    training_support: TrainingSupport | None = None,
    prior_value_range: tuple[float, float] | None = None,
    dense_value_diagnostic_steps: Sequence[int] = (),
    dense_value_diagnostic_points: int = 41,
    progress: Callable[[str], None] | None = None,
) -> list[ActiveLearningStep]:
    """Run config.acquisition_budget steps of the loop with strategy
    config.strategy (see `choose_acquisition`).

    Every strategy uses the same environment generator, so strategies run
    with the same seed start from the same random state.

    Args:
    ----------
    model, environment
        The model and the true system.

    initial_context : ExperimentalContext
        D_0.

    treatment, outcome : int
        The query.

    treatment_values : torch.Tensor
        The psi values [Q] (raw scale).

    config : ActiveLearningConfig
        Settings.

    generators : ActiveLearningGenerators
        Random streams.

    training_support : TrainingSupport | None
        If given, contexts outside it give a warning and the step's
        `outside_training_support` flag.

    prior_value_range : tuple[float, float] | None
        Raw range of value_domain "prior_range".

    dense_value_diagnostic_steps, dense_value_diagnostic_points
        Steps (1-based) at which the EIG strategy also runs the dense value
        diagnostic with this many values per node.

    progress : Callable[[str], None] | None
        Called with one line per step (e.g. print).
    """
    psi = torch.as_tensor(treatment_values, dtype=torch.float64)
    nodes = candidate_target_nodes(
        initial_context.num_variables, treatment, outcome,
        config.exclude_query_nodes,
    )  # fmt: skip
    if training_support is not None:
        problems = worst_case_violations(
            initial_context, config.acquisition_budget,
            config.rows_per_acquisition,
            num_possible_target_nodes(config.strategy, nodes),
            training_support,
        )  # fmt: skip
        for problem in problems:
            warnings.warn(
                f"active learning may leave the training support: {problem}",
                stacklevel=2,
            )
    truth = environment.true_query_distribution(treatment, outcome, psi)
    points = config.quadrature_points
    context = initial_context
    steps = []
    for step in range(1, config.acquisition_budget + 1):
        before = predict_query_on_context(model, context, treatment, outcome, psi)
        choice = choose_acquisition(
            model, environment, context, treatment, outcome, psi, config,
            generators, query_before=before, truth=truth,
            prior_value_range=prior_value_range,
            dense_value_diagnostic_points=(
                dense_value_diagnostic_points
                if step in dense_value_diagnostic_steps else None
            ),
        )  # fmt: skip
        chosen = choice.chosen
        rows = environment.run(chosen, generators.environment)
        context_after = context.append(rows, chosen.target_node)
        after = predict_query_on_context(
            model, context_after, treatment, outcome, psi
        )
        violations = (
            [] if training_support is None
            else context_violations(context_after, training_support)
        )  # fmt: skip
        for problem in violations:
            warnings.warn(
                f"step {step}: outside the training support: {problem}",
                stacklevel=2,
            )
        record = ActiveLearningStep(
            step=step,
            strategy=config.strategy,
            scores=choice.scores,
            chosen=chosen,
            choice_is_clear=choice.choice_is_clear,
            predicted_information_gain=choice.predicted_information_gain,
            real_rows=rows,
            query_before=before,
            query_after=after,
            realised_information_gain=float(
                kl_divergence_quadrature(after, before, points).mean()
            ),
            kl_to_truth_before=query_kl(truth, before, points),
            kl_to_truth_after=query_kl(truth, after, points),
            outside_training_support=bool(violations),
            support_violations=violations,
            dense_value_diagnostic=choice.dense_value_diagnostic,
        )
        steps.append(record)
        if progress is not None:
            predicted = record.predicted_information_gain
            progress(
                f"[{config.strategy}] step {step}: {chosen.describe()}"
                + (
                    f", EIG {predicted:.4f}"
                    + ("" if record.choice_is_clear else " (not clear-cut)")
                    if math.isfinite(predicted) else ""
                )
                + f", realised gain {record.realised_information_gain:.4f}, "
                f"KL to truth {record.kl_to_truth_after:.4f}"
            )  # fmt: skip
        context = context_after
    return steps
