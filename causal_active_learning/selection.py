"""The bi-level selection of the next acquisition: inner value search per
node, outer re-estimation, choice (ABCI Eq. 4.15)."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from causal_active_learning.acquisitions import (
    Acquisition,
    candidate_target_nodes,
)
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import UnivariateGaussianMixture
from causal_active_learning.fantasies import draw_fantasy_noise
from causal_active_learning.information_gain import (
    estimate_expected_information_gain,
    predict_query_on_context,
)
from causal_active_learning.protocols import CausalPredictiveModel
from causal_active_learning.value_search import (
    ExpectedInformationGainFunction,
    ValueEvaluation,
    ValueSearchResult,
    make_value_search,
)
from causal_active_learning.value_search.domain import ValueDomain, value_domains

if TYPE_CHECKING:
    from causal_active_learning.loop import ActiveLearningGenerators


# --------------------------------------------------------------------------- #
# Outer level
# --------------------------------------------------------------------------- #
@dataclass
class AcquisitionScore:
    """Outer-level result for one candidate.

    Attributes:
    ----------
    acquisition : Acquisition
        The node's recommended experiment.

    expected_information_gain, standard_error : float
        The re-evaluated estimate (fresh, `final_num_fantasies` fantasies).

    coherence_gap : float
        See `InformationGainEstimate.coherence_gap`.

    value_search_result : ValueSearchResult | None
        The inner-level result (None for the observational candidate).
    """

    acquisition: Acquisition
    expected_information_gain: float
    standard_error: float
    coherence_gap: float
    value_search_result: ValueSearchResult | None


def choice_is_clear(scores: list[AcquisitionScore]) -> bool:
    """False if the runner-up is within two combined standard errors of the
    best, sqrt(se_best^2 + se_second^2); True with a single candidate."""
    if len(scores) < 2:
        return True
    ranked = sorted(scores, key=lambda s: s.expected_information_gain)
    best, second = ranked[-1], ranked[-2]
    combined = math.sqrt(best.standard_error**2 + second.standard_error**2)
    gap = best.expected_information_gain - second.expected_information_gain
    return bool(gap > 2.0 * combined)


def reevaluate_and_select(
    evaluate_final: ExpectedInformationGainFunction,
    inner_results: dict[int, ValueSearchResult],
    extra_candidates: list[Acquisition] | None = None,
) -> tuple[Acquisition, list[AcquisitionScore]]:
    """Outer level: re-evaluate every node's recommended experiment (and the
    `extra_candidates`, e.g. an observational experiment) with
    `evaluate_final` (fresh noise, `final_num_fantasies`), and return the
    candidate with the largest re-evaluated EIG and all scores (in node
    order, extra candidates last)."""
    nodes = list(inner_results)
    candidates = [inner_results[n].acquisition for n in nodes]
    candidates += list(extra_candidates or [])
    estimates = evaluate_final(candidates)
    scores = [
        AcquisitionScore(
            acquisition=e.acquisition,
            expected_information_gain=e.expected_information_gain,
            standard_error=e.standard_error,
            coherence_gap=e.coherence_gap,
            value_search_result=inner_results[nodes[i]]
            if i < len(nodes)
            else None,
        )
        for i, e in enumerate(estimates)
    ]
    best = max(scores, key=lambda s: s.expected_information_gain)
    return best.acquisition, scores


# --------------------------------------------------------------------------- #
# A design step
# --------------------------------------------------------------------------- #
def search_intervention_values(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    generators: "ActiveLearningGenerators",
    domains: dict[int, ValueDomain],
    query_before: UnivariateGaussianMixture | None = None,
) -> dict[int, ValueSearchResult]:
    """Inner level: the most informative value of every node of `domains`
    (config.value_search), with one inner-level noise draw per node
    reused by every evaluation of that node (common random numbers)."""
    noise = draw_fantasy_noise(
        model, list(domains), config.num_fantasies, context.num_variables,
        config.rows_per_acquisition, generators.inner_fantasy,
    )  # fmt: skip

    def evaluate(candidates):
        return estimate_expected_information_gain(
            model, context, treatment, outcome, treatment_values, candidates,
            noise, config, query_before=query_before,
            kl_generator=generators.inner_fantasy,
        )  # fmt: skip

    optimiser = make_value_search(config)
    return optimiser.search(evaluate, domains, generators.bayesian_optimisation)


def select_next_acquisition(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    generators: "ActiveLearningGenerators",
    *,
    prior_value_range: tuple[float, float] | None = None,
    query_before: UnivariateGaussianMixture | None = None,
) -> tuple[Acquisition, list[AcquisitionScore]]:
    """One design step (the sequential wet-lab mode): the next acquisition
    for the query X_outcome | do(X_treatment = psi) on `context`.

    Inner level: per candidate node, the value maximising EIG
    (`search_intervention_values`). Outer level: every node's recommended
    acquisition (plus an observational one if enabled) re-estimated with
    config.final_num_fantasies fresh fantasies from the outer-level
    generator, with the coherence gap; the largest re-estimated EIG wins
    (`reevaluate_and_select`).

    Returns:
    ----------
    (the chosen acquisition, one `AcquisitionScore` per candidate).
    """
    psi = torch.as_tensor(treatment_values, dtype=torch.float64)
    nodes = candidate_target_nodes(
        context.num_variables, treatment, outcome, config.exclude_query_nodes
    )
    if not nodes and not config.include_observational_candidate:
        raise ValueError("there is no candidate acquisition")
    if query_before is None:
        query_before = predict_query_on_context(
            model, context, treatment, outcome, psi
        )
    domains = value_domains(context, nodes, config, prior_value_range)
    inner = search_intervention_values(
        model, context, treatment, outcome, psi, config, generators, domains,
        query_before,
    )  # fmt: skip
    extra = []
    if config.include_observational_candidate:
        extra.append(Acquisition(None, None, config.rows_per_acquisition))
    outer_nodes: list[int | None] = [*nodes]
    if extra:
        outer_nodes.append(None)
    outer_noise = draw_fantasy_noise(
        model, outer_nodes, config.final_num_fantasies, context.num_variables,
        config.rows_per_acquisition, generators.outer_fantasy,
    )  # fmt: skip

    def evaluate_final(candidates):
        return estimate_expected_information_gain(
            model, context, treatment, outcome, psi, candidates, outer_noise,
            config, query_before=query_before, compute_coherence_gap=True,
            kl_generator=generators.outer_fantasy,
        )  # fmt: skip

    return reevaluate_and_select(evaluate_final, inner, extra)


def dense_value_diagnostic(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    domains: dict[int, ValueDomain],
    num_values: int,
    generator: torch.Generator,
    query_before: UnivariateGaussianMixture | None = None,
) -> dict[int, list[ValueEvaluation]]:
    """EIG on `num_values` evenly spaced values of every node's domain, with
    config.final_num_fantasies fantasies (one noise per node from
    `generator`), to check that BO finds the optimum of exhaustive search."""
    noise = draw_fantasy_noise(
        model, list(domains), config.final_num_fantasies,
        context.num_variables, config.rows_per_acquisition, generator,
    )  # fmt: skip
    plan = [
        (node, float(u))
        for node, d in domains.items()
        for u in np.linspace(d.lower, d.upper, num_values)
    ]
    estimates = estimate_expected_information_gain(
        model, context, treatment, outcome, treatment_values,
        [domains[n].acquisition(u) for n, u in plan], noise, config,
        query_before=query_before, kl_generator=generator,
    )  # fmt: skip
    out: dict[int, list[ValueEvaluation]] = {n: [] for n in domains}
    for (node, u), e in zip(plan, estimates):
        out[node].append(
            ValueEvaluation(
                round=0,
                standardised_value=u,
                value=domains[node].to_value(u),
                expected_information_gain=e.expected_information_gain,
                standard_error=e.standard_error,
            )  # fmt: skip
        )
    return out
