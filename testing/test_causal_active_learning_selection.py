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
    reevaluate_and_select,
    search_intervention_values,
    select_next_acquisition,
)
from causal_active_learning.value_search import ValueSearchResult
from causal_active_learning.value_search.domain import value_domains

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    ConjugateStub,
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
