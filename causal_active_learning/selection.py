"""The bi-level selection of the next acquisition (ABCI Eq. 4.15, extended
with the observational candidate):

    inner level, for every intervenable node i:   v_i* in argmax_v EIG(i, v)
    outer level:   a* = argmax over {(i, v_i*)} union {observational batch}
                   of cost_adjusted_value(EIG(a), cost(a))

EIG here is the selected objective (`config.information_gain_objective`),
which may be negative for "entropy_reduction" (`cost_adjusted_value`).

Fantasy noise (`config.fantasy_noise`): "independent" (default) draws fresh
noise for every EIG evaluation, so each estimate is EIG(v) plus an error
independent across evaluations with variance equal to its squared standard
error, exactly the GP's observation model; "shared" reuses one noise draw per
node within a step (common random numbers). The noise of node i comes from a
generator that depends only on one draw from the inner-search generator and
i, so searching several nodes in lockstep equals searching each alone.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from causal_active_learning.acquisitions import (
    Acquisition,
    CandidateSet,
    InterventionalQuery,
    build_candidate_set,
)
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import UnivariateGaussianMixture
from causal_active_learning.information_gain import (
    estimate_expected_information_gain,
    predict_query_on_context,
)
from causal_active_learning.protocols import CausalPredictiveModel
from causal_active_learning.value_search import (
    ExpectedInformationGainFunction,
    ValueEvaluation,
    ValueSearchResult,
    _draw_base_seed,
    _node_generator,
    make_value_search,
)
from causal_active_learning.value_search.domain import ValueDomain, value_domains

if TYPE_CHECKING:
    from causal_active_learning.loop import ActiveLearningGenerators

OBSERVATIONAL_UNAVAILABLE_WARNING = (
    "the model cannot predict observational rows "
    "(supports_observational_prediction = False), so the expected information "
    "gain strategy skips the observational candidate"
)


@dataclass
class AcquisitionScore:
    """Outer-level result for one candidate.

    Attributes:
    ----------
    acquisition : Acquisition
        The candidate (a node at its best value, or observational rows).

    expected_information_gain, standard_error : float
        The re-estimated EIG (fresh noise, `final_num_fantasies`).

    coherence_gap : float
        See `InformationGainEstimate.coherence_gap`.

    value_search_result : ValueSearchResult | None
        The inner-level result (None for the observational candidate).

    cost : float
        The acquisition's cost; the outer level maximises
        `cost_adjusted_value(expected_information_gain, cost)`.

    objective : str
        The name of the selected objective (`expected_information_gain`).

    objective_values, objective_standard_errors : dict[str, float]
        Every objective's re-estimate and standard error (empty when not
        scored with the model).

    drift, entropy_shift : float
        See `InformationGainEstimate`.
    """

    acquisition: Acquisition
    expected_information_gain: float
    standard_error: float
    coherence_gap: float
    value_search_result: ValueSearchResult | None
    cost: float = 1.0
    objective: str = ""
    objective_values: dict[str, float] = field(default_factory=dict)
    objective_standard_errors: dict[str, float] = field(default_factory=dict)
    drift: float = float("nan")
    entropy_shift: float = float("nan")


def acquisition_cost(
    acquisition: Acquisition, config: ActiveLearningConfig
) -> float:
    """observational_cost or intervention_cost."""
    if acquisition.is_observational:
        return config.observational_cost
    return config.intervention_cost


def cost_adjusted_value(value: float, cost: float) -> float:
    """The value per unit cost the outer level ranks by:
        value / cost   if value >= 0,
        value * cost   if value < 0.
    Increasing in the value and decreasing in the cost on both sides, so a
    more expensive acquisition never ranks higher for the same value
    (dividing a negative value by a larger cost would bring it closer to
    zero, i.e. rank it higher). With cost 1 it is the value itself."""
    return value / cost if value >= 0 else value * cost


def _cost_adjusted_standard_error(score: "AcquisitionScore") -> float:
    """The standard error on the scale of `cost_adjusted_value`."""
    if score.expected_information_gain >= 0:
        return score.standard_error / score.cost
    return score.standard_error * score.cost


def _ranking_key(score: "AcquisitionScore") -> float:
    return cost_adjusted_value(score.expected_information_gain, score.cost)


def choice_is_clear(scores: list[AcquisitionScore]) -> bool:
    """False if the runner-up is within two combined standard errors of the
    winner on the cost-adjusted value (`cost_adjusted_value`, standard errors
    scaled the same way), sqrt(se_1^2 + se_2^2); True with a single
    candidate."""
    if len(scores) < 2:
        return True
    ranked = sorted(scores, key=_ranking_key)
    best, second = ranked[-1], ranked[-2]
    combined = math.sqrt(
        _cost_adjusted_standard_error(best) ** 2
        + _cost_adjusted_standard_error(second) ** 2
    )
    gap = _ranking_key(best) - _ranking_key(second)
    return bool(gap > 2.0 * combined)


def reevaluate_and_select(
    evaluate_final: ExpectedInformationGainFunction,
    value_search_results: dict[int, ValueSearchResult],
    extra_acquisitions: list[Acquisition] | None = None,
    config: ActiveLearningConfig | None = None,
) -> tuple[Acquisition, list[AcquisitionScore]]:
    """Outer level: re-estimate every node's best acquisition (and the
    `extra_acquisitions`, e.g. an observational batch) with `evaluate_final`
    and return the candidate with the largest cost-adjusted value
    (`cost_adjusted_value`; costs from `config`, all 1 without) and all
    scores (node order, extras last)."""
    nodes = list(value_search_results)
    acquisitions = [value_search_results[n].acquisition for n in nodes]
    acquisitions += list(extra_acquisitions or [])
    estimates = evaluate_final(acquisitions)
    scores = [
        AcquisitionScore(
            acquisition=e.acquisition,
            expected_information_gain=e.expected_information_gain,
            standard_error=e.standard_error,
            coherence_gap=e.coherence_gap,
            value_search_result=value_search_results[nodes[i]]
            if i < len(nodes)
            else None,
            cost=1.0
            if config is None
            else acquisition_cost(e.acquisition, config),
            objective=e.objective,
            objective_values=dict(e.objective_values),
            objective_standard_errors=dict(e.objective_standard_errors),
            drift=e.drift,
            entropy_shift=e.entropy_shift,
        )
        for i, e in enumerate(estimates)
    ]
    best = max(scores, key=_ranking_key)
    return best.acquisition, scores


def eig_candidate_set(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    config: ActiveLearningConfig,
) -> CandidateSet:
    """The candidate set of the EIG strategy: the observational candidate
    needs `model.supports_observational_prediction`."""
    return build_candidate_set(
        context.num_variables,
        config.intervenable_nodes,
        config.include_observational_candidate,
        bool(getattr(model, "supports_observational_prediction", False)),
    )


class _NoiseSupplier:
    """Fantasy noise per node for one step's value search: fresh for every
    evaluation ("independent") or one draw per node ("shared"), from per-node
    generators derived from one draw of `generator`."""

    def __init__(self, model, nodes, config, num_variables, generator):
        base = _draw_base_seed(generator)
        self.model = model
        self.config = config
        self.num_variables = num_variables
        self.generators = {n: _node_generator(base, n) for n in nodes}
        self.shared = {}
        if config.fantasy_noise == "shared":
            self.shared = {n: self._draw(n) for n in nodes}

    def _draw(self, node):
        return self.model.draw_fantasy_noise(
            self.num_variables,
            self.config.num_fantasies,
            self.config.rows_per_acquisition,
            self.generators[node],
        )

    def __call__(self, acquisitions):
        if self.config.fantasy_noise == "shared":
            return [self.shared[a.target_node] for a in acquisitions]
        return [self._draw(a.target_node) for a in acquisitions]


def search_intervention_values(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    query: InterventionalQuery,
    config: ActiveLearningConfig,
    generators: "ActiveLearningGenerators",
    domains: dict[int, ValueDomain],
    query_before: UnivariateGaussianMixture | None = None,
) -> dict[int, ValueSearchResult]:
    """Inner level: the most informative value of every node of `domains`
    (`config.value_search`), all nodes in lockstep."""
    supply = _NoiseSupplier(
        model,
        list(domains),
        config,
        context.num_variables,
        generators.inner_fantasy,
    )

    def evaluate(acquisitions):
        return estimate_expected_information_gain(
            model,
            context,
            query.treatment,
            query.outcome,
            query.treatment_values,
            acquisitions,
            supply(acquisitions),
            config,
            query_before=query_before,
            kl_generator=generators.inner_fantasy,
        )

    search = make_value_search(config)
    return search.search(evaluate, domains, generators.bayesian_optimisation)


def select_next_acquisition(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    query: InterventionalQuery,
    config: ActiveLearningConfig,
    generators: "ActiveLearningGenerators",
    *,
    training_value_range: tuple[float, float] | None = None,
    query_before: UnivariateGaussianMixture | None = None,
    warn_observational_unavailable: bool = True,
) -> tuple[Acquisition, list[AcquisitionScore]]:
    """One design step (the sequential wet-lab mode): the next acquisition
    for `query` on `context`, respecting `config.intervenable_nodes`. Needs
    no environment.

    Inner level: per intervenable node, the value maximising EIG
    (`search_intervention_values`). Outer level: every node's best
    acquisition, and an observational batch if included and the model can
    predict observational rows (otherwise one warning and the candidate is
    skipped), re-estimated with `final_num_fantasies` fresh fantasies from
    the outer-level generator, with the coherence gap; the largest
    cost-adjusted value (`cost_adjusted_value`) wins.

    Returns:
    ----------
    (the chosen acquisition, one `AcquisitionScore` per candidate).
    """
    candidates = eig_candidate_set(model, context, config)
    if (
        config.include_observational_candidate
        and not candidates.observational
        and warn_observational_unavailable
    ):
        warnings.warn(OBSERVATIONAL_UNAVAILABLE_WARNING, stacklevel=2)
    psi = query.treatment_values
    if query_before is None:
        query_before = predict_query_on_context(
            model, context, query.treatment, query.outcome, psi
        )
    domains = value_domains(
        context, candidates.intervenable_nodes, config, training_value_range
    )
    inner = search_intervention_values(
        model, context, query, config, generators, domains, query_before
    )
    extra = (
        [Acquisition.observational(config.rows_per_acquisition)]
        if candidates.observational
        else []
    )

    def evaluate_final(acquisitions):
        noise = [
            model.draw_fantasy_noise(
                context.num_variables,
                config.final_num_fantasies,
                config.rows_per_acquisition,
                generators.outer_fantasy,
            )
            for _ in acquisitions
        ]
        return estimate_expected_information_gain(
            model,
            context,
            query.treatment,
            query.outcome,
            psi,
            acquisitions,
            noise,
            config,
            query_before=query_before,
            compute_coherence_gap=True,
            kl_generator=generators.outer_fantasy,
        )

    return reevaluate_and_select(evaluate_final, inner, extra, config)


def dense_value_diagnostic(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    query: InterventionalQuery,
    config: ActiveLearningConfig,
    domains: dict[int, ValueDomain],
    num_values: int,
    generator: torch.Generator,
    query_before: UnivariateGaussianMixture | None = None,
) -> dict[int, list[ValueEvaluation]]:
    """EIG on `num_values` evenly spaced values of every node's domain with
    `final_num_fantasies` fantasies, once with independent noise (fresh per
    value) and once with shared noise (one draw per node), to check that the
    value search finds the optimum of exhaustive search. The evaluations
    carry their `noise_mode`."""
    out: dict[int, list[ValueEvaluation]] = {n: [] for n in domains}
    plan = [
        (node, float(u))
        for node, d in domains.items()
        for u in np.linspace(d.lower, d.upper, num_values)
    ]
    for mode in ("independent", "shared"):
        shared = {
            n: model.draw_fantasy_noise(
                context.num_variables,
                config.final_num_fantasies,
                config.rows_per_acquisition,
                generator,
            )
            for n in domains
        }
        noise = [
            shared[n]
            if mode == "shared"
            else model.draw_fantasy_noise(
                context.num_variables,
                config.final_num_fantasies,
                config.rows_per_acquisition,
                generator,
            )
            for n, _ in plan
        ]
        estimates = estimate_expected_information_gain(
            model,
            context,
            query.treatment,
            query.outcome,
            query.treatment_values,
            [domains[n].acquisition(u) for n, u in plan],
            noise,
            config,
            query_before=query_before,
            kl_generator=generator,
        )
        for (node, u), e in zip(plan, estimates):
            out[node].append(
                ValueEvaluation(
                    round=0,
                    standardised_value=u,
                    value=domains[node].to_value(u),
                    expected_information_gain=e.expected_information_gain,
                    standard_error=e.standard_error,
                    noise_mode=mode,
                )
            )
    return out
