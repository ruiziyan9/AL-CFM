"""Bi-level selection (plan tests 7, 10, 15, 16, 19; core, stub models)."""

import math
import sys
import warnings
from pathlib import Path

import pytest
import torch

from causal_active_learning import selection as selection_module
from causal_active_learning.acquisitions import Acquisition
from causal_active_learning.information_gain import InformationGainEstimate
from causal_active_learning.loop import ActiveLearningGenerators
from causal_active_learning.selection import (
    AcquisitionScore,
    choice_is_clear,
    cost_adjusted_value,
    reevaluate_and_select,
    search_intervention_values,
    select_next_acquisition,
)
from causal_active_learning.value_search import ValueSearchResult
from causal_active_learning.value_search.domain import value_domains

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    ConjugateStub,
    ShiftedWidenedStub,
    cheap_config,
    domain,
    empty_context,
    prior_context,
    query,
)


def _select(stub, config, context=None, seed=0, **kw):
    return select_next_acquisition(
        stub,
        context or prior_context(),
        query(),
        config,
        ActiveLearningGenerators.from_seed(seed),
        **kw,
    )


# 7. Observational candidate
def test_observational_candidate_chosen_when_most_informative():
    stub = ConjugateStub(observational_scale=10.0)
    config = cheap_config(
        value_domain="fixed_range",
        value_domain_fixed_range=(-0.5, 0.5),
        rows_per_acquisition=3,
    )
    chosen, scores = _select(stub, config)
    assert chosen == Acquisition.observational(3)
    assert scores[-1].acquisition.is_observational
    assert scores[-1].value_search_result is None
    costly = cheap_config(
        value_domain="fixed_range",
        value_domain_fixed_range=(-0.5, 0.5),
        rows_per_acquisition=3,
        observational_cost=1000.0,
    )
    chosen, _ = _select(stub, costly)
    assert not chosen.is_observational


def test_observational_candidate_unavailable_warns_and_is_skipped():
    stub = ConjugateStub(observational=False, observational_scale=10.0)
    with pytest.warns(UserWarning, match="observational candidate"):
        chosen, scores = _select(stub, cheap_config())
    assert not chosen.is_observational
    assert not any(s.acquisition.is_observational for s in scores)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _select(stub, cheap_config(include_observational_candidate=False))


# 10. Intervenable nodes decide the candidates
def test_treatment_chosen_only_when_intervenable():
    stub = ConjugateStub(strength={0: 1.0, 1: 0.3})
    fixed = dict(value_domain="fixed_range", value_domain_fixed_range=(-2.0, 2.0))
    config = cheap_config(include_observational_candidate=False, **fixed)
    chosen, _ = _select(stub, config)
    assert chosen.target_node == 0
    restricted = cheap_config(
        include_observational_candidate=False,
        intervenable_nodes=(1, 2, 3),
        **fixed,
    )
    chosen, scores = _select(stub, restricted)
    assert chosen.target_node == 1
    assert {s.acquisition.target_node for s in scores} == {1, 2, 3}


# 15. Noise modes
def _recorded_noise(monkeypatch, config):
    calls = []
    real = selection_module.estimate_expected_information_gain

    def recorder(
        model,
        context,
        treatment,
        outcome,
        values,
        acquisitions,
        noise,
        config_,
        **kwargs,
    ):
        calls.append((list(acquisitions), list(noise), kwargs))
        return real(
            model,
            context,
            treatment,
            outcome,
            values,
            acquisitions,
            noise,
            config_,
            **kwargs,
        )

    monkeypatch.setattr(
        selection_module, "estimate_expected_information_gain", recorder
    )
    _select(ConjugateStub(), config)
    return calls


def test_independent_noise_is_fresh_for_every_evaluation(monkeypatch):
    config = cheap_config(include_observational_candidate=False)
    calls = _recorded_noise(monkeypatch, config)
    inner, outer = calls[:-1], calls[-1]
    ids = [id(n) for _, noise, _ in inner for n in noise]
    assert len(ids) == len(set(ids))
    assert all(
        n.shape[0] == config.num_fantasies for _, ns, _ in inner for n in ns
    )
    assert all(n.shape[0] == config.final_num_fantasies for n in outer[1])
    assert outer[2]["compute_coherence_gap"]
    assert not {id(n) for n in outer[1]} & set(ids)


def test_shared_noise_is_one_draw_per_node(monkeypatch):
    config = cheap_config(
        fantasy_noise="shared", include_observational_candidate=False
    )
    calls = _recorded_noise(monkeypatch, config)
    per_node: dict = {}
    for acquisitions, noise, _ in calls[:-1]:
        for a, n in zip(acquisitions, noise):
            per_node.setdefault(a.target_node, set()).add(id(n))
    assert all(len(ids) == 1 for ids in per_node.values())
    assert len({next(iter(ids)) for ids in per_node.values()}) == len(per_node)
    inner_ids = set().union(*per_node.values())
    assert not {id(n) for n in calls[-1][1]} & inner_ids


# 16. Lockstep with the real estimator; only intervenable nodes searched
@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_lockstep_search_equals_searching_each_node_alone(mode):
    stub = ConjugateStub(strength={0: 0.5, 1: 1.0, 3: 0.2})
    config = cheap_config(fantasy_noise=mode)
    context = prior_context()
    domains = value_domains(context, [0, 1, 3], config)

    def search(nodes):
        return search_intervention_values(
            stub,
            context,
            query(),
            config,
            ActiveLearningGenerators.from_seed(7),
            {n: domains[n] for n in nodes},
        )

    together = search([0, 1, 3])
    for node in (0, 1, 3):
        alone = search([node])[node]
        assert [
            e.expected_information_gain for e in together[node].evaluations
        ] == [e.expected_information_gain for e in alone.evaluations]
        assert together[node].best_value == alone.best_value


# 19 and the outer level
def _result(node, value, surrogate):
    return ValueSearchResult(
        target_node=node,
        best_value=value,
        best_standardised_value=value,
        best_value_surrogate_mean=surrogate,
        evaluations=[],
        domain=domain(node),
    )


def test_outer_level_selects_by_the_reestimated_value():
    inner = {0: _result(0, 0.5, 0.9), 1: _result(1, -0.5, 0.2)}

    def evaluate_final(acquisitions):
        values = {0: 0.3, 1: 0.6}
        return [
            InformationGainEstimate(
                a, values[a.target_node], 0.01, torch.zeros(1)
            )
            for a in acquisitions
        ]

    chosen, scores = reevaluate_and_select(evaluate_final, inner)
    assert chosen.target_node == 1
    assert [s.value_search_result for s in scores] == [inner[0], inner[1]]


def test_choice_is_clear_flag():
    def score(value, se):
        return AcquisitionScore(
            Acquisition.intervention(0, 0.0), value, se, float("nan"), None
        )

    assert not choice_is_clear([score(0.50, 0.02), score(0.47, 0.02)])
    assert choice_is_clear([score(0.50, 0.01), score(0.40, 0.01)])
    assert choice_is_clear([score(0.5, 0.1)])


def test_scores_are_finite_with_coherence_gaps():
    chosen, scores = _select(
        ConjugateStub(), cheap_config(), context=empty_context()
    )
    best = max(scores, key=lambda s: s.expected_information_gain / s.cost)
    assert chosen == best.acquisition
    assert all(math.isfinite(s.coherence_gap) for s in scores)


# Entropy objective test 6: negative values through the whole step
def _negative_stub():
    """Every acquisition widens the query: w^2 = base + (value - optimum)^2
    with base > 2, so entropy_reduction = 0.5 log(1 / (0.5 w^2)) < 0;
    node 1 at value -0.4 is the least negative."""
    base = {0: 3.0, 1: 2.5, 2: 4.0, 3: 3.5}
    optimum = {0: 0.2, 1: -0.4, 2: 0.0, 3: 0.5}
    return ShiftedWidenedStub(
        base_rows=20,
        width_squared=lambda node, value: (
            base[node] + (value - optimum[node]) ** 2
        ),
    )


@pytest.mark.parametrize("search", ["bayesian_optimisation", "grid"])
def test_negative_values_run_the_full_step_and_choose_the_largest(search):
    config = cheap_config(
        value_search=search,
        include_observational_candidate=False,
        value_domain="fixed_range",
        value_domain_fixed_range=(-2.0, 2.0),
        bayesian_optimisation_iterations=4,
        grid_num_values=21,
    )
    assert config.information_gain_objective == "entropy_reduction"
    chosen, scores = _select(_negative_stub(), config, context=empty_context())
    values = [s.expected_information_gain for s in scores]
    assert all(math.isfinite(v) and v < 0 for v in values)
    best = max(scores, key=lambda s: s.expected_information_gain)
    assert chosen == best.acquisition
    assert chosen.target_node == 1
    assert abs(chosen.intervention_value + 0.4) < 0.3
    for s in scores:
        result = s.value_search_result
        assert all(
            math.isfinite(e.expected_information_gain)
            and e.expected_information_gain < 0
            for e in result.evaluations
        )
        if search == "bayesian_optimisation":
            assert result.surrogate is not None
            assert math.isfinite(result.best_value_surrogate_mean)
            ucb = [e.acquisition_value for e in result.evaluations[3:]]
            assert ucb and all(math.isfinite(u) for u in ucb)
    # the recommendation is near each node's optimum, not pulled to zero
    assert abs(scores[0].acquisition.intervention_value - 0.2) < 0.5


# Entropy objective test 7: the cost-adjusted ranking
def test_cost_adjusted_value():
    assert cost_adjusted_value(-1.0, 0.5) > cost_adjusted_value(-1.0, 2.0)
    assert cost_adjusted_value(1.0, 0.5) > cost_adjusted_value(1.0, 2.0)
    assert cost_adjusted_value(0.6, 2.0) == 0.3
    assert cost_adjusted_value(-0.6, 2.0) == -1.2
    for value in (-0.7, 0.0, 0.4):
        assert cost_adjusted_value(value, 1.0) == value
    # increasing in the value at any cost
    grid = [-2.0, -0.5, -1e-9, 0.0, 1e-9, 0.5, 2.0]
    for cost in (0.25, 1.0, 4.0):
        adjusted = [cost_adjusted_value(v, cost) for v in grid]
        assert adjusted == sorted(adjusted)


def _outer(values, costs, standard_error=0.01):
    """reevaluate_and_select over node 0, node 1 and observational rows with
    the given values and costs (observational, intervention)."""
    from causal_active_learning.config import ActiveLearningConfig

    inner = {0: _result(0, 0.5, 0.0), 1: _result(1, -0.5, 0.0)}

    def evaluate_final(acquisitions):
        return [
            InformationGainEstimate(
                a, values[a.target_node], standard_error, torch.zeros(1)
            )
            for a in acquisitions
        ]

    config = ActiveLearningConfig(
        1, observational_cost=costs[0], intervention_cost=costs[1]
    )
    return reevaluate_and_select(
        evaluate_final, inner, [Acquisition.observational(3)], config
    )


def test_cheaper_of_two_equal_negative_values_ranks_higher():
    values = {0: -0.4, 1: -0.9, None: -0.4}
    chosen, _ = _outer(values, (2.0, 1.0))
    assert chosen.target_node == 0  # the observational batch costs more
    chosen, _ = _outer(values, (0.5, 1.0))
    assert chosen.is_observational
    # value / cost would have ranked the more expensive one higher
    assert -0.4 / 2.0 > -0.4 / 1.0


def test_non_negative_values_keep_the_value_per_cost_ranking():
    values = {0: 0.3, 1: 0.5, None: 0.2}
    for costs in ((1.0, 1.0), (0.25, 1.0), (1.0, 3.0)):
        chosen, scores = _outer(values, costs)
        best = max(scores, key=lambda s: s.expected_information_gain / s.cost)
        assert chosen == best.acquisition
    assert _outer(values, (0.25, 1.0))[0].is_observational


def test_equal_costs_rank_by_the_plain_value():
    for values in (
        {0: -0.3, 1: -0.1, None: -0.2},
        {0: 0.3, 1: -0.1, None: 0.0},
        {0: 0.1, 1: 0.5, None: 0.2},
    ):
        for cost in (1.0, 3.0):
            chosen, scores = _outer(values, (cost, cost))
            best = max(scores, key=lambda s: s.expected_information_gain)
            assert chosen == best.acquisition


def test_choice_is_clear_with_costs_and_negative_values():
    def score(value, se, cost, node=0):
        return AcquisitionScore(
            Acquisition.intervention(node, 0.0),
            value,
            se,
            float("nan"),
            None,
            cost,
        )

    # equal negative values, different costs: adjusted -0.25 vs -1.0
    assert choice_is_clear([score(-0.5, 0.01, 0.5), score(-0.5, 0.01, 2.0, 1)])
    assert not choice_is_clear(
        [score(-0.50, 0.02, 1.0), score(-0.47, 0.02, 1.0, 1)]
    )
