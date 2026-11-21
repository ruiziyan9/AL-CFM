"""Strategies: how the next acquisition is chosen, given the context, the
candidates and the random streams. All share the loop."""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from causal_active_learning.acquisitions import (
    CandidateExperiment,
    candidate_target_nodes,
)
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentContext
from causal_active_learning.distributions import (
    QueryMixture,
    kl_divergence_quadrature,
)
from causal_active_learning.information_gain import predict_query_on_context
from causal_active_learning.protocols import (
    ExperimentEnvironment,
    PredictiveModel,
)
from causal_active_learning.selection import (
    CandidateScore,
    choice_is_clear,
    dense_value_diagnostic,
    optimise_intervention_values,
    select_next_experiment,
)
from causal_active_learning.value_search import ValueEvaluation
from causal_active_learning.value_search.domain import ValueDomain, value_domains

if TYPE_CHECKING:
    from causal_active_learning.loop import ActiveLearningGenerators


@dataclass
class StrategyChoice:
    """What a strategy decided in one step.

    Attributes:
    ----------
    chosen : CandidateExperiment
        The acquisition to run.

    scores : list[CandidateScore]
        Scores of the candidates (EIG strategies; empty for baselines).

    choice_is_clear : bool
        See `choice_is_clear` (True for the baselines).

    predicted_information_gain : float
        Re-estimated EIG of the chosen acquisition (NaN for baselines).

    dense_value_diagnostic : dict | None
        Dense value diagnostic, when requested.
    """

    chosen: CandidateExperiment
    scores: list[CandidateScore] = field(default_factory=list)
    choice_is_clear: bool = True
    predicted_information_gain: float = float("nan")
    dense_value_diagnostic: dict[int, list[ValueEvaluation]] | None = None


def query_kl(truth: QueryMixture, model: QueryMixture, points: int) -> float:
    """psi-averaged KL(truth || model) by quadrature."""
    return float(kl_divergence_quadrature(truth, model, points).mean())


def _uniform(domain: ValueDomain, generator: torch.Generator) -> float:
    u = float(torch.rand(1, generator=generator, dtype=torch.float64))
    return domain.lower + domain.width * u


def choose_acquisition(
    model: PredictiveModel,
    environment: ExperimentEnvironment,
    context: ExperimentContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    generators: "ActiveLearningGenerators",
    *,
    query_before: QueryMixture,
    truth: QueryMixture,
    prior_value_range: tuple[float, float] | None = None,
    dense_value_diagnostic_points: int | None = None,
) -> StrategyChoice:
    """The next acquisition of strategy `config.acquisition`:

    - "expected_information_gain": `select_next_experiment`;
    - "random": a uniformly random candidate node, value uniform over its
      domain (ABCI's RAND);
    - "observational": observational rows only (ABCI's OBS);
    - "treatment": always do(X_treatment), value uniform over its domain;
    - "exhaustive_true_gain" (evaluation only, not a method): the inner level
      for every node, then every node's acquisition run on the environment
      with a copy of the environment generator, keeping the one with the
      lowest query KL to the truth afterwards.
    """
    nodes = candidate_target_nodes(
        context.num_variables, treatment, outcome, config.exclude_query_nodes
    )
    acquisition = config.acquisition
    if acquisition == "expected_information_gain":
        chosen, scores = select_next_experiment(
            model, context, treatment, outcome, treatment_values, config,
            generators, prior_value_range=prior_value_range,
            query_before=query_before,
        )  # fmt: skip
        predicted = next(
            s.expected_information_gain for s in scores if s.candidate == chosen
        )
        diagnostic = None
        if dense_value_diagnostic_points:
            domains = value_domains(context, nodes, config, prior_value_range)
            diagnostic = dense_value_diagnostic(
                model, context, treatment, outcome, treatment_values, config,
                domains, dense_value_diagnostic_points, generators.diagnostic,
                query_before=query_before,
            )  # fmt: skip
        return StrategyChoice(
            chosen, scores, choice_is_clear(scores), predicted, diagnostic
        )
    if acquisition == "random":
        index = int(
            torch.randint(len(nodes), (1,), generator=generators.baseline)
        )
        node = nodes[index]
        domain = value_domains(context, [node], config, prior_value_range)[node]
        return StrategyChoice(
            domain.candidate(_uniform(domain, generators.baseline))
        )
    if acquisition == "observational":
        return StrategyChoice(
            CandidateExperiment(None, None, config.rows_per_experiment)
        )
    if acquisition == "treatment":
        domain = value_domains(context, [treatment], config, prior_value_range)[
            treatment
        ]
        return StrategyChoice(
            domain.candidate(_uniform(domain, generators.baseline))
        )
    # exhaustive_true_gain
    domains = value_domains(context, nodes, config, prior_value_range)
    inner = optimise_intervention_values(
        model, context, treatment, outcome, treatment_values, config,
        generators, domains, query_before,
    )  # fmt: skip
    best_kl = math.inf
    best: CandidateExperiment | None = None
    scores = []
    for node, result in inner.items():
        trial = environment.run_experiment(
            result.candidate, copy.deepcopy(generators.environment)
        )
        after = predict_query_on_context(
            model, context.append(trial, node), treatment, outcome,
            treatment_values,
        )  # fmt: skip
        kl = query_kl(truth, after, config.quadrature_points)
        scores.append(
            CandidateScore(
                candidate=result.candidate,
                expected_information_gain=result.best_value_posterior_mean,
                standard_error=float("nan"),
                coherence_gap=float("nan"),
                inner_result=result,
            )
        )
        if kl < best_kl:
            best_kl, best = kl, result.candidate
    assert best is not None
    return StrategyChoice(best, scores)


def num_possible_target_nodes(acquisition: str, nodes: Sequence[int]) -> int:
    """How many distinct nodes a strategy can intervene on."""
    return {"observational": 0, "treatment": 1}.get(acquisition, len(nodes))
