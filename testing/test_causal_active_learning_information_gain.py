"""The EIG estimator against closed forms (plan tests 3, 4, 6; core)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_active_learning.acquisitions import Acquisition
from causal_active_learning.distributions import UnivariateGaussianMixture
from causal_active_learning.information_gain import (
    check_objective_identities,
    estimate_expected_information_gain,
    information_gain_objectives,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    PSI,
    ConjugateStub,
    ContextBlindStub,
    ShiftedWidenedStub,
    cheap_config,
    empty_context,
    prior_context,
)

OBJECTIVES = ("entropy_reduction", "kl_divergence", "mixture_entropy_reduction")


def _estimate(
    stub, acquisitions, S, D=4, seed=0, objective="entropy_reduction", **kw
):
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
        cheap_config(quadrature_points=512, information_gain_objective=objective),
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
    # KL form: under the conjugate stub H(p_s) does not depend on the
    # fantasy, so the entropy form is exact even with independent rows.
    S, v = 2000, 1.5
    exact = ConjugateStub().exact_information_gain(1, v, num_rows=3)
    independent = _estimate(
        ConjugateStub(independent_rows=True),
        [Acquisition.intervention(1, v, 3)],
        S,
        seed=3,
        objective="kl_divergence",
    )[0]
    gap = independent.expected_information_gain - exact
    assert abs(gap) > 6 * independent.standard_error + 1e-3


def test_standard_error_matches_the_spread_of_repeated_estimates():
    # KL form: the entropy form has no spread under the conjugate stub.
    stub = ConjugateStub()
    acquisition = [Acquisition.intervention(1, 1.0, 3)]
    estimates, errors = [], []
    for seed in range(40):
        e = _estimate(
            stub, acquisition, 32, seed=seed, objective="kl_divergence"
        )[0]
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


# Entropy objective test 2: identities on synthetic mixtures
def _random_mixtures(*lead, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return UnivariateGaussianMixture(
        torch.randn(*lead, K, generator=g, dtype=torch.float64),
        0.3 + torch.rand(*lead, K, generator=g, dtype=torch.float64),
        torch.log_softmax(
            torch.randn(*lead, K, generator=g, dtype=torch.float64), -1
        ),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_objective_identities_on_synthetic_mixtures(seed):
    C, S, Q = 3, 16, 4
    after = _random_mixtures(C, S, Q, seed=seed)
    before = _random_mixtures(Q, K=3, seed=seed + 100)
    o = information_gain_objectives(
        after, before, cheap_config(quadrature_points=1024)
    )
    means = {n: t.mean(-1) for n, t in o.terms.items()}
    mixture = means["mixture_entropy_reduction"]
    torch.testing.assert_close(
        means["kl_divergence"] - mixture, o.drift, rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        means["entropy_reduction"] - mixture, o.entropy_shift, rtol=0, atol=1e-6
    )
    check_objective_identities(means, o.drift, o.entropy_shift)
    assert bool((o.drift > 0).all())  # incoherent mixtures
    with pytest.raises(AssertionError, match="drift"):
        check_objective_identities(means, o.drift + 1e-3, o.entropy_shift)


# Entropy objective test 3: a coherent model
@pytest.mark.parametrize("R", [1, 3])
def test_objectives_agree_for_the_coherent_stub(R):
    stub, v, S = ConjugateStub(), 1.5, 2000
    exact = stub.exact_information_gain(1, v, num_rows=R)
    e = _estimate(
        stub,
        [Acquisition.intervention(1, v, R)],
        S,
        seed=5,
        compute_coherence_gap=True,
        check_identities=True,
    )[0]
    assert e.objective == "entropy_reduction"
    assert set(e.objective_values) == set(OBJECTIVES)
    for name in OBJECTIVES:
        error = e.objective_standard_errors[name]
        assert abs(e.objective_values[name] - exact) < 4 * error + 2e-3, name
    # H(p_s) does not depend on the fantasy outcome under the stub
    assert e.objective_values["entropy_reduction"] == pytest.approx(
        exact, abs=1e-6
    )
    # drift is second order in the Monte Carlo error of p_bar, the entropy
    # shift first order (here the error of mixture_entropy_reduction)
    assert abs(e.drift) < 3e-3
    mixture_error = e.objective_standard_errors["mixture_entropy_reduction"]
    assert abs(e.entropy_shift) < 4 * mixture_error + 2e-3
    assert e.expected_information_gain == e.objective_values["entropy_reduction"]


# Entropy objective test 4: a model that is not coherent
@pytest.mark.parametrize("w2", [3.0, 1.2])
def test_objectives_of_the_shifted_widened_stub(w2):
    stub = ShiftedWidenedStub(width=math.sqrt(w2))
    exact = stub.closed_forms()
    S = 1000
    acquisition = Acquisition.intervention(1, 0.5, 1)
    estimates = {}
    for name in OBJECTIVES:
        g = torch.Generator().manual_seed(0)
        noise = [stub.draw_fantasy_noise(4, S, 1, g)]
        estimates[name] = estimate_expected_information_gain(
            stub,
            empty_context(rows=20),
            0,
            2,
            PSI,
            [acquisition],
            noise,
            cheap_config(information_gain_objective=name, quadrature_points=512),
            check_identities=True,
        )[0]
    e = estimates["entropy_reduction"]
    for name in OBJECTIVES:  # the same fantasies give the same values
        assert estimates[name].objective_values == e.objective_values
        assert (
            estimates[name].expected_information_gain
            == (e.objective_values[name])
        )
    values, errors = e.objective_values, e.objective_standard_errors
    assert values["kl_divergence"] > 0
    assert values["entropy_reduction"] == pytest.approx(
        exact["entropy_reduction"], abs=1e-6
    )
    assert errors["entropy_reduction"] < 1e-12
    assert (values["entropy_reduction"] < 0) == (w2 * 0.5 > 1.0)
    assert abs(values["kl_divergence"] - exact["kl_divergence"]) < (
        4 * errors["kl_divergence"] + 1e-3
    )
    # p_bar is the S-component empirical mixture, not the Gaussian limit
    assert (
        abs(
            values["mixture_entropy_reduction"]
            - exact["mixture_entropy_reduction"]
        )
        < 4 * errors["mixture_entropy_reduction"] + 0.01
    )
    assert e.drift == pytest.approx(exact["drift"], abs=0.02)
    assert e.entropy_shift == pytest.approx(exact["entropy_shift"], abs=0.02)
    assert e.drift > 0.1  # the shift alone gives shift^2 / 2 = 0.245
