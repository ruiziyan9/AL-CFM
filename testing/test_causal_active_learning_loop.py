"""The loop, contexts, config validation and the support check (core)."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_active_learning.acquisitions import CandidateExperiment
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentContext
from causal_active_learning.loop import (
    ActiveLearningGenerators,
    run_active_learning,
)
from causal_active_learning.support_check import TrainingSupport

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    PSI,
    ConjugateStub,
    StubEnvironment,
    empty_context,
)
from test_causal_active_learning_selection import cheap_config


def test_context_append():
    context = empty_context(rows=15, D=5)
    rows = torch.randn(2, 5, dtype=torch.float64)
    after = context.append(rows, 3)
    assert after.values.shape == (17, 5)
    assert torch.equal(after.values[:15], context.values)
    assert after.intervened[15:, 3].all() and after.intervened.sum() == 2
    assert not context.append(rows, None).intervened.any()
    with pytest.raises(ValueError):
        ExperimentContext(torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        ExperimentContext(
            torch.tensor([[float("nan"), 0.0]]),
            torch.zeros(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        ExperimentContext(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))


def test_reproducible_and_real_rows_independent_of_acquisition_settings():
    stub = ConjugateStub()

    def run(**settings):
        return run_active_learning(
            stub, StubEnvironment(), empty_context(), 0, 2, PSI,
            cheap_config(**settings), ActiveLearningGenerators.from_seed(3),
        )  # fmt: skip

    first, second = run(), run()
    for a, b in zip(first, second):
        assert a.chosen == b.chosen and torch.equal(a.real_rows, b.real_rows)
    random_a = run(acquisition="random")
    random_b = run(
        acquisition="random", bayesian_optimisation_iterations=5, num_fantasies=12
    )
    for a, b in zip(random_a, random_b):
        assert torch.equal(a.real_rows, b.real_rows)


def test_training_support_warning_and_flag():
    support = TrainingSupport((10, 21), 11, 2)
    stub = ConjugateStub()
    with pytest.warns(UserWarning, match="training support"):
        steps = run_active_learning(
            stub, StubEnvironment(), empty_context(rows=20), 0, 2, PSI,
            cheap_config(acquisition="random"),
            ActiveLearningGenerators.from_seed(0), training_support=support,
        )  # fmt: skip
    assert not steps[0].outside_training_support
    assert steps[1].outside_training_support
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_active_learning(
            stub, StubEnvironment(), empty_context(rows=20), 0, 2, PSI,
            cheap_config(acquisition="random"),
            ActiveLearningGenerators.from_seed(0),
            training_support=TrainingSupport((10, 100), 11, 2),
        )  # fmt: skip


@pytest.mark.parametrize(
    "settings",
    [
        {"acquisition": "greedy"},
        {"value_optimiser": "random_search"},
        {"kl_estimator": "exact"},
        {"value_domain": "anywhere"},
        {"experiment_budget": 0},
        {"num_fantasies": 1},
        {"rows_per_experiment": 0},
        {"inference_batch_size": 0},
        {"bayesian_optimisation_initial_points": 1},
        {"value_domain_half_width": 0.0},
    ],
)
def test_config_validation(settings):
    base = {"experiment_budget": 3}
    base.update(settings)
    with pytest.raises(ValueError):
        ActiveLearningConfig(**base)


def test_candidate_validation():
    with pytest.raises(ValueError):
        CandidateExperiment(1, None)
    with pytest.raises(ValueError):
        CandidateExperiment(None, 1.0)
    assert CandidateExperiment(None, None).is_observational
    assert np.isfinite(1.0)
