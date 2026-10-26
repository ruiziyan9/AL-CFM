"""MOTIF3 graph prior, exact question-class table and graph-independence (S3)."""

import dataclasses
import inspect
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.linear_gaussian_prior import LinearGaussianPrior
from CITNP.datasets.mixed_regime_generator import (
    MixedRegimeConfig,
    generate_task,
    sample_design,
    sample_graph,
)
from CITNP.datasets.motifs import (
    MOTIF_CLASSES,
    MOTIF_SIZES,
    motif_class,
    motif_dags,
)
from CITNP.evaluation.oracle import LinGaussOracle, all_dags, dag_key, mec_key
from CITNP.evaluation.tasks import empirical_graph_prior, motif_graph_prior
from CITNP.prior_diagnostics.stats import QUESTION_CLASSES, question_class_table

CONFIGS = (
    Path(__file__).resolve().parents[1] / "CITNP" / "experiments" / "configs"
)
PROBS = dict(
    complete=0.30, chain=0.20, fork=0.20, collider=0.15, single=0.10, empty=0.05
)
BASE = dict(
    function_generator="linear_iw",
    alpha0=1.5,
    num_variables=[3],
    graph_type=["MOTIF3"],
    graph_degrees=[1],
    motif_probs=PROBS,
    normalise=False,
    n_ctx_range=(64, 96),
    n_qry=6,
    qry_mode="grouped",
)


def test_class_sizes():
    dags = motif_dags()
    assert {c: len(v) for c, v in dags.items()} == MOTIF_SIZES
    assert [len(dags[c]) for c in MOTIF_CLASSES] == [1, 6, 6, 3, 3, 6]
    assert all(motif_class(g) == c for c, v in dags.items() for g in v)
    assert sum(len(v) for v in dags.values()) == len(all_dags(3))


def test_motif_validation():
    cfg = MixedRegimeConfig(**BASE)
    assert cfg.motif_probs["fork"] == 0.2
    with pytest.raises(ValueError):
        MixedRegimeConfig(**{**BASE, "motif_probs": {**PROBS, "empty": 0.5}})
    with pytest.raises(ValueError):
        MixedRegimeConfig(
            **{**BASE, "motif_probs": {**PROBS, "empty": -0.05, "chain": 0.30}}
        )
    with pytest.raises(ValueError):
        MixedRegimeConfig(**{**BASE, "motif_probs": {"star": 1.0}})
    with pytest.raises(ValueError):
        MixedRegimeConfig(**{**BASE, "num_variables": [3, 4]})
    with pytest.raises(ValueError):
        MixedRegimeConfig(**{**BASE, "motif_probs": None})
    with pytest.raises(ValueError):
        MixedRegimeConfig(**{**BASE, "graph_type": ["ER"]})


def test_motif_graph_prior_closed_form():
    cfg = MixedRegimeConfig(**BASE)
    lp = motif_graph_prior(cfg)
    assert len(lp) == 25
    assert np.isclose(sum(np.exp(v) for v in lp.values()), 1.0, atol=1e-12)
    for g in all_dags(3):
        c = motif_class(g)
        assert np.isclose(np.exp(lp[dag_key(g)]), PROBS[c] / MOTIF_SIZES[c])
    # empirical_graph_prior dispatches to it
    assert empirical_graph_prior(cfg) == lp
    zero = MixedRegimeConfig(**{**BASE, "motif_probs": {"complete": 1.0}})
    lp0 = motif_graph_prior(zero)
    assert sum(np.isfinite(v) for v in lp0.values()) == 6


@pytest.mark.slow
def test_sample_graph_matches_closed_form():
    cfg = MixedRegimeConfig(**BASE)
    lp = motif_graph_prior(cfg)
    n = 100_000
    np.random.seed(0)
    counts = defaultdict(int)
    for _ in range(n):
        g, info = sample_graph(cfg, 3)
        counts[dag_key(g.astype(int))] += 1
    keys = list(lp)
    obs = np.array([counts[k] for k in keys])
    assert obs.sum() == n
    exp = np.array([np.exp(lp[k]) * n for k in keys])
    assert stats.chisquare(obs, exp).pvalue > 1e-3


def _legacy_oracle_d3_prior():
    """Exact graph prior of oracle_d3.yaml (ER, s0 uniform in {1, 2, 3}): single
    1/3, complete 1/3, and the three 2-edge classes 1/9 each; uniform in class."""
    probs = dict(
        single=1 / 3, chain=1 / 9, fork=1 / 9, collider=1 / 9, complete=1 / 3
    )
    return {
        dag_key(g): np.log(
            probs.get(motif_class(g), 0.0) / MOTIF_SIZES[motif_class(g)]
        )
        if probs.get(motif_class(g), 0.0) > 0
        else -np.inf
        for g in all_dags(3)
    }


def test_question_class_table_oracle_d3():
    table = question_class_table(_legacy_oracle_d3_prior())
    expect = {
        "unconfounded effect": 16 / 54,
        "confounded effect": 3 / 54,
        "pure confounding": 2 / 54,
        "reverse": 19 / 54,
        "unrelated": 14 / 54,
    }
    assert set(table) == set(QUESTION_CLASSES)
    for k, v in expect.items():
        assert abs(table[k] - v) < 1e-12


def test_question_class_table_motif_config():
    cfg = load_config(CONFIGS / "oracle_d3_iw_motif.yaml")
    table = question_class_table(motif_graph_prior(cfg))
    expect = {
        "unconfounded effect": 1 / 3,
        "confounded effect": 1 / 20,
        "pure confounding": 1 / 15,
        "reverse": 23 / 60,
        "unrelated": 1 / 6,
    }
    assert abs(sum(table.values()) - 1) < 1e-12
    for k, v in expect.items():
        assert abs(table[k] - v) < 1e-12


def test_design_does_not_depend_on_graph():
    assert "graph" not in inspect.signature(sample_design).parameters
    cfg = MixedRegimeConfig(**BASE)
    g1, g2 = motif_dags()["complete"][0], motif_dags()["empty"][0]
    for seed in range(3):
        a = generate_task(cfg, seed, graph=g1, permute=False)
        b = generate_task(cfg, seed, graph=g2, permute=False)
        assert a.regime_targets == b.regime_targets
        for k in ("regime_sizes", "do_dist", "alpha", "K", "has_obs",
                  "qry_targets", "group_treatments", "n_groups", "group_types"):  # fmt: skip
            assert a.meta[k] == b.meta[k], k


def test_within_mec_posterior_ratio_equals_prior_ratio():
    cfg = dataclasses.replace(
        load_config(CONFIGS / "oracle_d3_iw_a15.yaml"), k_max=0
    )
    dags = all_dags(3)
    lp = np.random.default_rng(0).normal(size=len(dags))
    graph_prior = {dag_key(g): float(v) for g, v in zip(dags, lp)}
    prior = LinearGaussianPrior.from_config(cfg)
    assert prior.is_iw_compatible
    for i in range(5):
        task = generate_task(cfg, i, num_variables=3)
        assert not task.ctx_int.any()  # observation-only
        orc = LinGaussOracle(3, log_graph_prior=graph_prior, prior=prior).fit(
            task.ctx_x.astype(np.float64), task.ctx_int.astype(bool)
        )
        for a in range(len(dags)):
            for b in range(a + 1, len(dags)):
                if mec_key(dags[a]) == mec_key(dags[b]):
                    d_post = orc.log_post[a] - orc.log_post[b]
                    assert abs(d_post - (lp[a] - lp[b])) < 1e-8
