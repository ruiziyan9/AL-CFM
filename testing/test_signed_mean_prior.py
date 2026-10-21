"""Tests for the signed-mean weight prior (weight_mean) of the
linearnonidentifiable generator and the exact oracle:
    w | s2 ~ N(weight_mean * s, eta * s2 * I),  s in {+1, -1}^p uniform.
"""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from CITNP.datasets.mixed_regime_generator import (
    SOURCE_ADHOC,
    Design,
    MixedRegimeConfig,
    generate_task,
    make_key,
)
from CITNP.evaluation.oracle import (
    NODE_A,
    NODE_B,
    LinGaussOracle,
    dag_key,
    node_posterior,
)

BASE = dict(
    function_generator="linearnonidentifiable",
    num_variables=[3],
    graph_type=["ER"],
    graph_degrees={3: [1, 2, 3]},
    normalise=False,
    n_ctx_range=(64, 128),
    n_qry=48,
    qry_mode="grouped",
)
TASK_ARRAYS = ["ctx_x", "ctx_int", "ctx_regime", "qry_x", "qry_int", "qry_tgt", "graph"]


def _data(n=200, seed=0):
    """Observational rows of a 3-variable chain-like system, no interventions."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x[:, 1] += 1.5 * x[:, 0]
    x[:, 2] += -0.8 * x[:, 0] + 1.2 * x[:, 1]
    return x, np.zeros_like(x, dtype=bool)


# --------------------------------------------------------------------------- #
# Config and generator
# --------------------------------------------------------------------------- #
def test_weight_mean_validation():
    assert MixedRegimeConfig(**BASE).weight_mean == 0.0
    assert MixedRegimeConfig(**BASE, weight_mean=1).weight_mean == 1.0
    with pytest.raises(ValueError):
        MixedRegimeConfig(**BASE, weight_mean=-0.5)
    with pytest.raises(ValueError):
        MixedRegimeConfig(
            **{**BASE, "function_generator": "linearfixedstd"}, weight_mean=1.0
        )


@pytest.mark.parametrize("i", range(5))
def test_default_weight_mean_gives_identical_tasks(i):
    key = make_key(SOURCE_ADHOC, 0, 0, i)
    a = generate_task(MixedRegimeConfig(**BASE), key)
    b = generate_task(MixedRegimeConfig(**BASE, weight_mean=0.0), key)
    for k in TASK_ARRAYS:
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)
    assert a.meta == b.meta


def test_signs_recorded_and_weights_shifted():
    key = make_key(SOURCE_ADHOC, 0, 0, 3)
    _, off = generate_task(MixedRegimeConfig(**BASE), key, return_debug=True)
    _, on = generate_task(
        MixedRegimeConfig(**BASE, weight_mean=1.0), key, return_debug=True
    )
    for rec in off["mechanisms"].values():
        assert rec["mechanism"].sign is None
    n_edges = 0
    for rec in on["mechanisms"].values():
        f = rec["mechanism"]
        if rec["parents"]:
            assert set(np.unique(f.sign)) <= {-1.0, 1.0}
            assert f.sign.shape == f.weight.shape == (len(rec["parents"]), 1)
            n_edges += len(rec["parents"])
        else:
            assert f.sign is None
    assert n_edges > 0


# --------------------------------------------------------------------------- #
# Oracle: default unchanged, continuity at 0
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("parents", [[], [0], [0, 1]])
def test_oracle_default_unchanged(parents):
    x, m = _data()
    pa = np.array(parents, dtype=int)
    a = node_posterior(x, m, 2, pa)
    b = node_posterior(x, m, 2, pa, weight_mean=0.0)
    assert (a.root, a.a_n, a.b_n, a.log_ml) == (b.root, b.a_n, b.b_n, b.log_ml)
    for f in ("m_n", "V_n"):
        if getattr(a, f) is None:
            assert getattr(b, f) is None
        else:
            np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    assert b.comp_m is None and b.comp_b is None and b.comp_logw is None

    orc0 = LinGaussOracle(3).fit(x, m)
    orc1 = LinGaussOracle(3, weight_mean=0.0).fit(x, m)
    np.testing.assert_array_equal(orc0.log_post, orc1.log_post)
    for r0, r1 in zip(
        orc0.predictive_components(0, 2, n_mc=200, seed=1),
        orc1.predictive_components(0, 2, n_mc=200, seed=1),
    ):
        np.testing.assert_array_equal(r0, r1)  # same random numbers


@pytest.mark.parametrize("parents", [[0], [0, 1]])
def test_oracle_continuous_at_zero(parents):
    x, m = _data()
    pa = np.array(parents, dtype=int)
    ref = node_posterior(x, m, 2, pa)
    eps = node_posterior(x, m, 2, pa, weight_mean=1e-8)
    assert abs(eps.log_ml - ref.log_ml) < 1e-6
    w = np.exp(eps.comp_logw)
    np.testing.assert_allclose(w @ eps.comp_m, ref.m_n, atol=1e-6)
    np.testing.assert_allclose(eps.comp_b, ref.b_n, rtol=1e-6)
    np.testing.assert_allclose(
        LinGaussOracle(3, weight_mean=1e-8).fit(x, m).graph_post,
        LinGaussOracle(3).fit(x, m).graph_post,
        atol=1e-6,
    )


def test_oracle_predictions_continuous_at_zero():
    # The mixture path draws a sign pattern per Monte Carlo sample, so the
    # random stream differs from weight_mean = 0: predictions agree only up to
    # Monte Carlo error, not to 1e-6 (the exact parts are tested above).
    x, m = _data(n=60)
    t = np.linspace(-2, 2, 9)
    y = 0.5 * t
    lp0 = LinGaussOracle(3).fit(x, m).logpdf(0, 2, t, y, n_mc=4000, seed=0)
    lp1 = LinGaussOracle(3, weight_mean=1e-8).fit(x, m).logpdf(
        0, 2, t, y, n_mc=4000, seed=0
    )
    assert np.abs(lp1 - lp0).mean() < 0.05


# --------------------------------------------------------------------------- #
# Conjugacy: log marginal likelihood against an independent formula
# --------------------------------------------------------------------------- #
def _brute_log_ml(X, y, m, eta):
    """log of the average over sign patterns s of the NIG marginal likelihood
    with prior mean m * s, written as the multivariate Student-t marginal:
    y ~ t_{2a}(X mu, (b / a) (I + eta X X^T))."""
    a, b = NODE_A, NODE_B
    n, p = X.shape
    shape = (b / a) * (np.eye(n) + eta * X @ X.T)
    lps = [
        multivariate_t(loc=X @ (m * np.array(s)), shape=shape, df=2 * a).logpdf(y)
        for s in itertools.product([1.0, -1.0], repeat=p)
    ]
    return logsumexp(lps) - p * np.log(2.0), np.array(lps)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("m, eta", [(0.8, 1.0), (1.5, 2.5)])
def test_log_ml_matches_brute_force_mixture(p, m, eta):
    rng = np.random.default_rng(10 + p)
    n = 15
    x = rng.normal(size=(n, 3))
    x[:, 2] = x[:, :p] @ rng.choice([-1.0, 1.0], size=p) * m + rng.normal(
        scale=0.5, size=n
    )
    mask = np.zeros_like(x, dtype=bool)
    post = node_posterior(x, mask, 2, np.arange(p), eta=eta, weight_mean=m)
    brute, lps = _brute_log_ml(x[:, :p], x[:, 2], m, eta)
    np.testing.assert_allclose(post.log_ml, brute, rtol=1e-9, atol=1e-9)
    # Component posterior weights are the normalised component evidences.
    np.testing.assert_allclose(post.comp_logw, lps - logsumexp(lps), atol=1e-9)
    assert post.comp_m.shape == (2**p, p) and post.comp_b.shape == (2**p,)


def test_intervened_rows_are_dropped_for_the_node():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 3))
    mask = np.zeros_like(x, dtype=bool)
    mask[20:, 2] = True  # rows 20-29: do(X2), not generated by X2's mechanism
    full = node_posterior(x, mask, 2, np.array([0]), weight_mean=1.0)
    kept = node_posterior(
        x[:20], mask[:20], 2, np.array([0]), weight_mean=1.0
    )
    assert full.log_ml == pytest.approx(kept.log_ml, rel=1e-12)


# --------------------------------------------------------------------------- #
# Oracle prior predictive vs the generator
# --------------------------------------------------------------------------- #
# Fixed complete DAG 0 -> 1, 0 -> 2, 1 -> 2 (node 2 has p = 2 parents).
GRAPH = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int8)


def _oracle_prior_moments(weight_mean, n_mc=20000):
    """Prior-predictive mean and variance of each node (zero data rows) from
    the oracle's parameter draws: X = A e, A = (I - B^T)^{-1}."""
    orc = LinGaussOracle(
        3, log_graph_prior={dag_key(GRAPH): 0.0}, weight_mean=weight_mean
    ).fit(np.zeros((0, 3)), np.zeros((0, 3), dtype=bool))
    g_i = int(np.argmax(orc.log_post))
    assert dag_key(orc.dags[g_i]) == dag_key(GRAPH)
    B, s2 = orc._sample_params(g_i, n_mc, np.random.default_rng(0))
    A = np.linalg.inv(np.eye(3)[None] - np.transpose(B, (0, 2, 1)))
    var = np.einsum("mjk,mk->mj", A**2, s2)  # [n_mc, 3] per-draw variances
    return var.mean(0), var.std(0) / np.sqrt(n_mc)


def test_oracle_prior_predictive_matches_generator():
    cfg = MixedRegimeConfig(
        "linearnonidentifiable", [3], ["ER"], {3: [2, 3]},
        normalise=False, weight_mean=1.0, n_ctx_range=(64, 64), n_qry=4,
    )  # fmt: skip
    design = Design(
        num_variables=3,
        regime_targets=[()],
        regime_sizes=[64],
        do_dist={"kind": "normal", "scale": 1.0},
        qry_targets=[()] * 4,
    )
    n_tasks = 3000
    mean_t, sq_t = [], []
    for i in range(n_tasks):
        task = generate_task(
            cfg, make_key(SOURCE_ADHOC, 0, 9, i), graph=GRAPH, design=design,
            permute=False,
        )  # fmt: skip
        np.testing.assert_array_equal(task.graph, GRAPH)
        mean_t.append(task.ctx_x.mean(0))
        sq_t.append((task.ctx_x.astype(np.float64) ** 2).mean(0))
    mean_t, sq_t = np.array(mean_t), np.array(sq_t)

    # Prior-predictive mean is 0 (no intercepts, zero-mean noise and signs).
    se_mean = mean_t.std(0) / np.sqrt(n_tasks)
    assert (np.abs(mean_t.mean(0)) < 4 * se_mean).all(), (mean_t.mean(0), se_mean)

    # Prior-predictive variance E[X_j^2] against the oracle's draws.
    gen_var, gen_se = sq_t.mean(0), sq_t.std(0) / np.sqrt(n_tasks)
    orc_var, orc_se = _oracle_prior_moments(1.0)
    tol = 4 * np.sqrt(gen_se**2 + orc_se**2)
    assert (np.abs(gen_var - orc_var) < tol).all(), (gen_var, orc_var, tol)

    # The check has power: the zero-mean prior predicts clearly smaller
    # variances for the nodes with parents.
    zero_var, zero_se = _oracle_prior_moments(0.0)
    gap = orc_var[1:] - zero_var[1:]
    assert (gap > 4 * np.sqrt(orc_se[1:] ** 2 + zero_se[1:] ** 2)).all()
