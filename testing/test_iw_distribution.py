"""The linear_iw generator matches the theory it was derived from (S2.2).

Slow: 1e5 prior draws per case. Under Sigma ~ IW(2 alpha0 + D - 1, 2 beta I),
beta = 0.5, every marginal variance is IG(alpha0, beta) and every correlation
has density proportional to (1 - rho^2)^(alpha0 - 1), i.e. (rho + 1) / 2 ~
Beta(alpha0, alpha0). Colliders are deliberately not tested: the marginal
variance law is not guaranteed there."""

import numpy as np
import pytest
from scipy import stats

from CITNP.datasets.functions_generator import LinearIWFunctionGenerator
from CITNP.datasets.linear_gaussian_prior import LinearGaussianPrior
from CITNP.prior_diagnostics.linear_gaussian import population_covariance

pytestmark = pytest.mark.slow

N_DRAWS = 100_000
BETA = 0.5
GRAPHS = {
    "complete": [(0, 1), (0, 2), (1, 2)],
    "empty": [],
    "single": [(0, 1)],
    "chain": [(0, 1), (1, 2)],
    "fork": [(1, 0), (1, 2)],
}


def _adj(edges):
    g = np.zeros((3, 3), dtype=int)
    for i, j in edges:
        g[i, j] = 1
    return g


def _population_covs(graph, alpha0, seed=0):
    np.random.seed(seed)
    prior = LinearGaussianPrior.iw(alpha0)
    gen = LinearIWFunctionGenerator(3, 1, prior=prior)
    out = np.empty((N_DRAWS, 3, 3))
    for n in range(N_DRAWS):
        fdict = gen.generate_functions(graph)
        for j, f in fdict.items():
            f(np.zeros((1, int(graph[:, j].sum()))))  # one call draws the weights
        out[n] = population_covariance(graph, fdict)
    return out


@pytest.fixture(scope="module")
def covs():
    cache = {}

    def get(name, alpha0):
        if (name, alpha0) not in cache:
            cache[(name, alpha0)] = _population_covs(_adj(GRAPHS[name]), alpha0)
        return cache[(name, alpha0)]

    return get


def _ks_variances(S, alpha0):
    ig = stats.invgamma(a=alpha0, scale=BETA)
    for j in range(3):
        assert stats.kstest(S[:, j, j], ig.cdf).pvalue > 1e-3


@pytest.mark.parametrize("alpha0", [1.0, 2.5])
def test_complete_dag_variances_and_correlations(covs, alpha0):
    S = covs("complete", alpha0)
    _ks_variances(S, alpha0)
    beta_law = stats.beta(alpha0, alpha0)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        rho = S[:, i, j] / np.sqrt(S[:, i, i] * S[:, j, j])
        assert stats.kstest((rho + 1) / 2, beta_law.cdf).pvalue > 1e-3


@pytest.mark.parametrize("alpha0", [1.0, 2.5])
@pytest.mark.parametrize("name", ["empty", "single", "chain", "fork"])
def test_indegree_le_1_variances(covs, name, alpha0):
    _ks_variances(covs(name, alpha0), alpha0)
