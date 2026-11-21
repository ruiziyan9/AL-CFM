"""The EIG estimator against closed forms (core, stub models)."""

import sys
from pathlib import Path

import pytest
import torch

from causal_active_learning.acquisitions import Acquisition
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.fantasies import draw_fantasy_noise
from causal_active_learning.information_gain import (
    estimate_expected_information_gain,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    PSI,
    ConjugateStub,
    ContextBlindStub,
    empty_context,
)


def _config(**kw):
    settings = dict(acquisition_budget=1, quadrature_points=512)
    settings.update(kw)
    return ActiveLearningConfig(**settings)


def _noise(stub, nodes, S, rows=1, seed=0):
    return draw_fantasy_noise(
        stub, nodes, S, 4, rows, torch.Generator().manual_seed(seed)
    )


def test_context_blind_query_gives_zero_information():
    stub = ContextBlindStub()
    candidates = [Acquisition(1, 1.5), Acquisition(3, -0.5)]
    estimates = estimate_expected_information_gain(
        stub, empty_context(), 0, 2, PSI, candidates, _noise(stub, [1, 3], 8),
        _config(), compute_coherence_gap=True,
    )  # fmt: skip
    for e in estimates:
        assert e.expected_information_gain == 0.0
        assert e.standard_error == 0.0
        assert e.coherence_gap == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("v", [0.5, 1.5, -2.5])
def test_conjugate_information_gain_matches_closed_form(v):
    stub = ConjugateStub()
    S = 2000
    estimate = estimate_expected_information_gain(
        stub, empty_context(), 0, 2, PSI, [Acquisition(1, v)],
        _noise(stub, [1], S, seed=3), _config(), compute_coherence_gap=True,
    )[0]  # fmt: skip
    exact = stub.exact_information_gain(v, PSI)
    assert abs(estimate.expected_information_gain - exact) < (
        4 * estimate.standard_error + 1e-3
    )
    assert estimate.coherence_gap < 2e-3
    uninformative = estimate_expected_information_gain(
        stub, empty_context(), 0, 2, PSI, [Acquisition(3, v)],
        _noise(stub, [3], 8), _config(),
    )[0]  # fmt: skip
    assert uninformative.expected_information_gain == 0.0
