"""Outer level and the design step (core, stub models)."""

import math
import sys
from pathlib import Path

import torch

from causal_active_learning import selection as selection_module
from causal_active_learning.acquisitions import (
    CandidateExperiment,
    candidate_target_nodes,
)
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.fantasies import draw_fantasy_noise
from causal_active_learning.information_gain import InformationGainEstimate
from causal_active_learning.loop import ActiveLearningGenerators
from causal_active_learning.selection import (
    CandidateScore,
    choice_is_clear,
    reevaluate_and_select,
    select_next_experiment,
)
from causal_active_learning.value_search import ValueOptimisationResult

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    PSI,
    ConjugateStub,
    domain,
    empty_context,
)


def cheap_config(**kw):
    settings = dict(
        experiment_budget=2, num_fantasies=8, final_num_fantasies=16,
        quadrature_points=256, bayesian_optimisation_iterations=2,
        gaussian_process_restarts=2, acquisition_grid_points=128,
    )  # fmt: skip
    settings.update(kw)
    return ActiveLearningConfig(**settings)


def _inner(node, value, surrogate):
    return ValueOptimisationResult(
        target_node=node, best_value=value, best_standardised_value=value,
        best_value_posterior_mean=surrogate, evaluations=[], domain=domain(node),
    )  # fmt: skip


def test_outer_level_selects_by_the_reevaluated_value():
    inner = {0: _inner(0, 1.0, 0.9), 1: _inner(1, -1.0, 0.2)}

    def evaluate_final(candidates):
        values = {0: 0.3, 1: 0.6}
        return [
            InformationGainEstimate(
                c, values[c.target_node], 0.01, torch.zeros(1)
            )
            for c in candidates
        ]

    chosen, scores = reevaluate_and_select(evaluate_final, inner)
    assert chosen.target_node == 1
    assert [s.inner_result for s in scores] == [inner[0], inner[1]]
    assert choice_is_clear(scores)


def test_choice_is_clear_flag():
    def score(value, se):
        return CandidateScore(
            CandidateExperiment(0, 0.0), value, se, float("nan"), None
        )

    assert not choice_is_clear([score(0.50, 0.02), score(0.47, 0.02)])
    assert choice_is_clear([score(0.50, 0.01), score(0.40, 0.01)])
    assert choice_is_clear([score(0.5, 0.1)])


def test_outer_level_uses_fresh_noise_and_final_fantasies(monkeypatch):
    stub = ConjugateStub()
    config = cheap_config(final_num_fantasies=24)
    calls = []
    real = selection_module.estimate_expected_information_gain

    def recorder(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        selection_module, "estimate_expected_information_gain", recorder
    )
    seed = [5, 6]
    chosen, scores = select_next_experiment(
        stub, empty_context(), 0, 2, PSI, config,
        ActiveLearningGenerators.from_seed(seed),
    )  # fmt: skip
    final_noise = calls[-1][6]
    expected = draw_fantasy_noise(
        stub, [1, 3, None], 24, 4, 1,
        ActiveLearningGenerators.from_seed(seed).outer_fantasy,
    )  # fmt: skip
    for node in (1, 3):
        assert torch.equal(final_noise[node], expected[node])
    assert all(n.shape[0] == config.num_fantasies for n in calls[0][6].values())
    best = max(scores, key=lambda s: s.expected_information_gain)
    assert chosen == best.candidate
    assert all(math.isfinite(s.coherence_gap) for s in scores)


def test_only_the_informative_node_is_chosen():
    stub = ConjugateStub()
    chosen, scores = select_next_experiment(
        stub, empty_context(), 0, 2, PSI, cheap_config(),
        ActiveLearningGenerators.from_seed(0),
    )  # fmt: skip
    assert chosen.target_node == 1
    uninformative = next(s for s in scores if s.candidate.target_node == 3)
    assert uninformative.expected_information_gain == 0.0


def test_candidate_exclusions():
    assert candidate_target_nodes(5, 2, 4, True) == [0, 1, 3]
    assert candidate_target_nodes(5, 2, 4, False) == [0, 1, 2, 3]
