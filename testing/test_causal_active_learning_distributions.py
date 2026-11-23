"""Univariate Gaussian mixtures and their KL divergence (core)."""

import math

import pytest
import torch

from causal_active_learning.distributions import (
    UnivariateGaussianMixture,
    average_mixture,
    average_mixture_divergences,
    entropy_quadrature,
    kl_divergence_monte_carlo,
    kl_divergence_quadrature,
)

DOUBLE = torch.float64


def _gaussian_kl(m1, s1, m2, s2):
    return math.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5


@pytest.mark.parametrize(
    "m1, s1, m2, s2",
    [
        (0.3, 0.8, -0.4, 1.7),
        (0.0, 0.01, 0.5, 20.0),  # narrow p inside a broad q
        (0.0, 5.0, 1.0, 0.05),  # q much narrower than p
    ],
)
def test_quadrature_kl_matches_the_closed_form(m1, s1, m2, s2):
    p = UnivariateGaussianMixture.gaussian(torch.tensor([m1]), torch.tensor([s1]))
    q = UnivariateGaussianMixture.gaussian(torch.tensor([m2]), torch.tensor([s2]))
    value = float(kl_divergence_quadrature(p, q, 1024))
    exact = _gaussian_kl(m1, s1, m2, s2)
    assert math.isfinite(value)
    assert value == pytest.approx(exact, rel=1e-6, abs=1e-6)


def test_quadrature_kl_matches_monte_carlo_for_mixtures():
    g = torch.Generator().manual_seed(0)
    p = UnivariateGaussianMixture(
        torch.tensor([[-1.0, 1.5]], dtype=DOUBLE),
        torch.tensor([[0.5, 0.8]], dtype=DOUBLE),
        torch.log(torch.tensor([[0.3, 0.7]], dtype=DOUBLE)),
    )
    q = UnivariateGaussianMixture(
        torch.tensor([[0.0, 2.0, -2.0]], dtype=DOUBLE),
        torch.tensor([[1.0, 0.6, 1.2]], dtype=DOUBLE),
        torch.log(torch.tensor([[0.5, 0.3, 0.2]], dtype=DOUBLE)),
    )
    quadrature = float(kl_divergence_quadrature(p, q, 1024)[0])
    draws = torch.stack(
        [kl_divergence_monte_carlo(p, q, 4000, g)[0] for _ in range(10)]
    )
    error = float(draws.std() / math.sqrt(len(draws)))
    assert abs(float(draws.mean()) - quadrature) < 4 * error + 1e-3
    assert quadrature > 0


def test_mixture_moments():
    mixture = UnivariateGaussianMixture(
        torch.tensor([[-1.0, 1.0]], dtype=DOUBLE),
        torch.tensor([[1.0, 1.0]], dtype=DOUBLE),
        torch.log(torch.tensor([[0.5, 0.5]], dtype=DOUBLE)),
    )
    mean, variance = mixture.mean_and_variance()
    assert float(mean) == pytest.approx(0.0)
    assert float(variance) == pytest.approx(2.0)


def _random_mixtures(*lead, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return UnivariateGaussianMixture(
        torch.randn(*lead, K, generator=g, dtype=DOUBLE),
        0.3 + torch.rand(*lead, K, generator=g, dtype=DOUBLE),
        torch.log_softmax(torch.randn(*lead, K, generator=g, dtype=DOUBLE), -1),
    )


# Entropy test 1
@pytest.mark.parametrize("sd", [0.01, 0.7, 30.0])
def test_entropy_of_a_gaussian_matches_the_closed_form(sd):
    mixture = UnivariateGaussianMixture.gaussian(
        torch.tensor([1.3]), torch.tensor([sd])
    )
    value = float(entropy_quadrature(mixture, 1024)[0])
    assert value == pytest.approx(
        0.5 * math.log(2 * math.pi * math.e * sd**2), abs=1e-6
    )


def test_entropy_of_a_mixture_matches_a_fine_grid():
    mixture = UnivariateGaussianMixture(
        torch.tensor([[-2.0, 1.5]], dtype=DOUBLE),
        torch.tensor([[0.4, 1.1]], dtype=DOUBLE),
        torch.log(torch.tensor([[0.35, 0.65]], dtype=DOUBLE)),
    )
    y = torch.linspace(-30, 30, 600_001, dtype=DOUBLE)
    density = sum(
        w / (s * math.sqrt(2 * math.pi)) * torch.exp(-0.5 * ((y - m) / s) ** 2)
        for m, s, w in ((-2.0, 0.4, 0.35), (1.5, 1.1, 0.65))
    )
    reference = float(-torch.trapezoid(density * torch.log(density), y))
    value = float(entropy_quadrature(mixture, 1024)[0])
    assert value == pytest.approx(reference, abs=1e-6)


def test_entropy_batch_shapes():
    mixtures = _random_mixtures(3, 4, K=5)
    value = entropy_quadrature(mixtures, 256)
    assert value.shape == (3, 4) and value.dtype == DOUBLE
    assert torch.equal(value[1], entropy_quadrature(mixtures[1], 256))
    single = UnivariateGaussianMixture.gaussian(torch.zeros(()), torch.ones(()))
    assert entropy_quadrature(single, 256).shape == ()


def test_average_mixture_divergences_match_direct_quadrature():
    C, S, Q = 2, 5, 3
    mixtures = _random_mixtures(C, S, Q, seed=1)
    reference = _random_mixtures(Q, K=2, seed=2)
    terms = average_mixture_divergences(mixtures, reference, 1024)
    assert terms.kl_to_average.shape == (C, S, Q)
    assert terms.average_entropy.shape == (C, Q)
    assert terms.average_kl_to_reference.shape == (C, Q)
    average = average_mixture(mixtures, axis=1)
    torch.testing.assert_close(
        terms.average_entropy,
        entropy_quadrature(average, 1024),
        rtol=0,
        atol=1e-9,
    )
    torch.testing.assert_close(
        terms.average_kl_to_reference,
        kl_divergence_quadrature(average, reference, 1024),
        rtol=0,
        atol=1e-9,
    )
    # each p_s on its own grid instead of p_bar's
    direct = kl_divergence_quadrature(mixtures, average[:, None], 1024)
    torch.testing.assert_close(terms.kl_to_average, direct, rtol=0, atol=1e-7)
    assert bool((terms.kl_to_average <= math.log(S) + 1e-9).all())
