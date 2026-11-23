"""The EIG estimator against closed forms (plan tests 3, 4, 6; core)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_active_learning.acquisitions import Acquisition
from causal_active_learning.information_gain import (
    estimate_expected_information_gain,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    PSI,
    ConjugateStub,
    ContextBlindStub,
    cheap_config,
    prior_context,
)


def _estimate(stub, acquisitions, S, D=4, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    noise = [stub.draw_fantasy_noise(D, S, a.num_rows, g) for a in acquisitions]
    return estimate_expected_information_gain(
        stub,
        prior_context(D=D),
        0,
        2,
        PSI,
        acquisitions,
        noise,
        cheap_config(quadrature_points=512),
        **kw,
    )


@pytest.mark.parametrize("D", [3, 5, 8])
@pytest.mark.parametrize("R", [1, 3])
def test_context_blind_query_gives_zero_information(D, R):
    stub = ContextBlindStub()
    acquisitions = [
        Acquisition.intervention(1, 1.5, R),
        Acquisition.intervention(D - 1, -0.5, R),
        Acquisition.observational(R),
    ]
    for e in _estimate(stub, acquisitions, 8, D=D, compute_coherence_gap=True):
        assert e.expected_information_gain == 0.0
        assert e.standard_error == 0.0
        assert e.coherence_gap == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("R", [1, 3])
@pytest.mark.parametrize("v", [0.5, -1.5])
def test_conjugate_information_gain_matches_closed_form(R, v):
    stub = ConjugateStub()
    S = 2000
    e = _estimate(
        stub,
        [Acquisition.intervention(1, v, R)],
        S,
        seed=3,
        compute_coherence_gap=True,
    )[0]
    exact = stub.exact_information_gain(1, v, num_rows=R)
    assert abs(e.expected_information_gain - exact) < 4 * e.standard_error + 1e-3
    assert e.coherence_gap < 3e-3
    obs = _estimate(
        stub,
        [Acquisition.observational(R)],
        S,
        seed=4,
        compute_coherence_gap=True,
    )[0]
    exact_obs = stub.observational_information_gain(num_rows=R)
    assert abs(obs.expected_information_gain - exact_obs) < (
        4 * obs.standard_error + 2e-3
    )
    assert obs.coherence_gap < 3e-3
    uninformative = _estimate(stub, [Acquisition.intervention(3, v, R)], 8)[0]
    assert uninformative.expected_information_gain == 0.0


def test_independent_rows_disagree_with_the_closed_form_at_three_rows():
    S, v = 2000, 1.5
    exact = ConjugateStub().exact_information_gain(1, v, num_rows=3)
    independent = _estimate(
        ConjugateStub(independent_rows=True),
        [Acquisition.intervention(1, v, 3)],
        S,
        seed=3,
    )[0]
    gap = independent.expected_information_gain - exact
    assert abs(gap) > 6 * independent.standard_error + 1e-3


def test_standard_error_matches_the_spread_of_repeated_estimates():
    stub = ConjugateStub()
    acquisition = [Acquisition.intervention(1, 1.0, 3)]
    estimates, errors = [], []
    for seed in range(40):
        e = _estimate(stub, acquisition, 32, seed=seed)[0]
        estimates.append(e.expected_information_gain)
        errors.append(e.standard_error)
    ratio = float(np.std(estimates, ddof=1)) / float(np.mean(errors))
    assert 0.6 < ratio < 1.6, ratio
    assert all(math.isfinite(x) and x >= 0 for x in estimates)


# Entropy objective test 5
def test_default_objective_is_entropy_reduction_and_names_are_validated():
    from causal_active_learning.config import (
        INFORMATION_GAIN_OBJECTIVES,
        ActiveLearningConfig,
    )

    assert ActiveLearningConfig(1).information_gain_objective == (
        "entropy_reduction"
    )
    for name in INFORMATION_GAIN_OBJECTIVES:
        assert ActiveLearningConfig(1, information_gain_objective=name)
    for bad in ("kl", "entropy", "mutual_information", ""):
        with pytest.raises(ValueError, match="information_gain_objective"):
            ActiveLearningConfig(1, information_gain_objective=bad)
