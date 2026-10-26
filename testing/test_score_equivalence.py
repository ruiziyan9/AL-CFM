"""Score equivalence of the IW prior (S2.1).

Under an IW-compatible prior, Markov-equivalent DAGs have identical marginal
likelihoods on observational data. The legacy prior breaks this for nodes with
two parents (complete DAG, colliders)."""

from collections import defaultdict

import numpy as np
import pytest

from CITNP.datasets.linear_gaussian_prior import LinearGaussianPrior
from CITNP.evaluation.oracle import all_dags, mec_key, node_posterior

D = 3
DAGS = all_dags(D)


def _classes():
    out = defaultdict(list)
    for i, g in enumerate(DAGS):
        out[mec_key(g)].append(i)
    return out


def _datasets(n_sets=5):
    """Observational datasets from arbitrary Gaussians (not from the prior)."""
    rng = np.random.default_rng(123)
    out = []
    for i in range(n_sets):
        n = (20, 200)[i % 2]
        A = rng.normal(size=(D, D))
        out.append((rng.normal(size=(n, D)) @ A.T + rng.normal(size=D), n))
    return out


def _log_ml(x, mask, prior):
    """Total log marginal likelihood of every DAG in DAGS."""
    out = []
    for g in DAGS:
        total = 0.0
        for j in range(D):
            pa = np.where(g[:, j])[0]
            total += node_posterior(x, mask, j, pa, prior=prior).log_ml
        out.append(total)
    return np.array(out)


def test_mec_counts():
    classes = _classes()
    assert len(DAGS) == 25 and len(classes) == 11
    sizes = sorted(len(v) for v in classes.values())
    # empty + 3 colliders: 1; 3 single edges: 2; 3 chain/fork classes: 3; complete: 6
    assert sizes == [1] * 4 + [2] * 3 + [3] * 3 + [6]
    empty = [i for i, g in enumerate(DAGS) if g.sum() == 0]
    assert len(classes[mec_key(DAGS[empty[0]])]) == 1
    complete = [i for i, g in enumerate(DAGS) if g.sum() == 3]
    assert len(complete) == 6 and len({mec_key(DAGS[i]) for i in complete}) == 1
    for i, g in enumerate(DAGS):  # colliders are singletons
        if g.sum() == 2 and (g.sum(0) == 2).any():
            assert len(classes[mec_key(g)]) == 1


@pytest.mark.parametrize("alpha0", [1.0, 1.5, 2.5])
def test_iw_prior_is_score_equivalent(alpha0):
    prior = LinearGaussianPrior.iw(alpha0)
    classes = _classes()
    for x, n in _datasets():
        lml = _log_ml(x, np.zeros_like(x, dtype=bool), prior)
        for members in classes.values():
            np.testing.assert_allclose(lml[members], lml[members[0]], atol=1e-8)


def test_legacy_prior_negative_control():
    prior = LinearGaussianPrior.legacy()
    classes = _classes()
    spread_complete = 0.0
    for x, n in _datasets():
        lml = _log_ml(x, np.zeros_like(x, dtype=bool), prior)
        for key, members in classes.items():
            spread = lml[members].max() - lml[members].min()
            if len(members) == 6:  # complete DAG
                spread_complete = max(spread_complete, spread)
            else:  # in-degree <= 1 everywhere (or a singleton class)
                assert spread < 1e-8
    assert spread_complete > 1e-6


@pytest.mark.parametrize(
    "prior", [LinearGaussianPrior.iw(2.5), LinearGaussianPrior.legacy()]
)
def test_interventional_rows_run(prior):
    """With interventional rows nothing is asserted about equality."""
    x, _ = _datasets()[0]
    mask = np.zeros_like(x, dtype=bool)
    mask[-5:, 0] = True
    assert np.isfinite(_log_ml(x, mask, prior)).all()
