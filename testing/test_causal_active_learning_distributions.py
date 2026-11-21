"""Univariate Gaussian mixtures and their KL divergence (core)."""

import math

import pytest
import torch

from causal_active_learning.distributions import (
    UnivariateGaussianMixture,
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
