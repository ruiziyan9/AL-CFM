"""Tests for the weight_scale / p_k0 / ER_CC generator options, the oracle's
eta and the oracle_d3_strong.yaml prior."""

import numpy as np
import pytest

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_ADHOC,
    MixedRegimeConfig,
    generate_task,
    make_key,
)
from CITNP.evaluation.oracle import LinGaussOracle, node_posterior

STRONG = "CITNP/experiments/configs/oracle_d3_strong.yaml"
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


def test_strong_config_loads():
    cfg = load_config(STRONG)
    assert cfg.weight_scale == 4.0 and cfg.p_k0 == 0.15 and "ER_CC" in cfg.graph_type


def test_explicit_defaults_are_identical():
    key = make_key(SOURCE_ADHOC, 0, 0, 5)
    a = generate_task(MixedRegimeConfig(**BASE), key)
    b = generate_task(MixedRegimeConfig(**BASE, weight_scale=1.0, p_k0=None), key)
    assert np.array_equal(a.ctx_x, b.ctx_x) and np.array_equal(a.qry_x, b.qry_x)


def test_p_k0_share_of_observation_only_contexts():
    cfg = load_config(STRONG)
    K = [generate_task(cfg, make_key(SOURCE_ADHOC, 0, 0, i)).meta["K"] for i in range(600)]
    assert abs(np.mean(np.array(K) == 0) - 0.15) < 0.05


def test_er_cc_graphs_have_a_common_cause():
    cfg = MixedRegimeConfig(**{**BASE, "graph_type": ["ER_CC"], "graph_degrees": {3: [2, 3]}})
    for i in range(100):
        g = generate_task(cfg, make_key(SOURCE_ADHOC, 0, 0, i)).graph
        assert g.sum(axis=1).max() >= 2


def test_er_cc_rejects_too_few_edges():
    with pytest.raises(ValueError):
        MixedRegimeConfig(**{**BASE, "graph_type": ["ER_CC"]})  # degree 1 allowed


def test_weight_scale_reaches_mechanisms():
    cfg = load_config(STRONG)
    _, debug = generate_task(cfg, make_key(SOURCE_ADHOC, 0, 0, 1), return_debug=True)
    assert all(r["mechanism"].eta == 4.0 for r in debug["mechanisms"].values())


def test_weight_scale_rejected_for_other_families():
    with pytest.raises(ValueError):
        MixedRegimeConfig(**{**BASE, "function_generator": "linearfixedstd"}, weight_scale=2.0)


def test_oracle_eta_default_unchanged_and_used():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    x[:, 1] += 1.5 * x[:, 0]
    m = np.zeros_like(x, dtype=bool)
    pa = np.array([0])
    base = node_posterior(x, m, 1, pa).log_ml
    assert node_posterior(x, m, 1, pa, eta=1.0).log_ml == base
    assert node_posterior(x, m, 1, pa, eta=4.0).log_ml != base
    orc = LinGaussOracle(3, eta=4.0).fit(x, m)
    assert orc.eta == 4.0 and np.isclose(orc.graph_post.sum(), 1.0)
