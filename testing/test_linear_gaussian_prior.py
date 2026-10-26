"""Tests for LinearGaussianPrior (S1.1) and the generator keys / wiring (S1.2-S1.3)."""

import warnings

import numpy as np
import pytest

from CITNP.datasets import linear_gaussian_prior as lgp
from CITNP.datasets.functions_generator import (
    LinearNIG,
    LinearNIGLegacy,
    LinearNonIdentifiable,
)
from CITNP.datasets.linear_gaussian_prior import (
    LinearGaussianPrior,
    is_linear_gaussian,
)
from CITNP.datasets.mixed_regime_generator import REGISTRY, MixedRegimeConfig

BASE = dict(
    num_variables=[3],
    graph_type=["ER"],
    graph_degrees={3: [1, 2, 3]},
    normalise=False,
    n_ctx_range=(64, 64),
    n_qry=6,
    qry_mode="grouped",
)


def test_node_shape_legacy():
    p = LinearGaussianPrior.legacy()
    assert [p.node_shape(k) for k in (0, 1, 2)] == [2.5, 3.0, 3.0]
    assert p.ig_scale == 0.5 and p.eta == 1.0 and p.weight_mean == 0.0


def test_node_shape_iw():
    p = LinearGaussianPrior.iw(2.5)
    assert [p.node_shape(k) for k in (0, 1, 2)] == [2.5, 3.0, 3.5]
    assert p.eta == 1.0
    assert LinearGaussianPrior.iw(1.5, ig_scale=2.0).eta == 0.25


@pytest.mark.parametrize("alpha0", [1.0, 1.5, 2.5])
def test_iw_is_iw_compatible(alpha0):
    assert LinearGaussianPrior.iw(alpha0).is_iw_compatible


@pytest.mark.parametrize(
    "prior",
    [
        LinearGaussianPrior.legacy(),
        LinearGaussianPrior.legacy(eta=9.0),
        LinearGaussianPrior.legacy(weight_mean=0.5),
    ],
)
def test_legacy_is_not_iw_compatible(prior):
    assert not prior.is_iw_compatible


@pytest.mark.parametrize(
    "kw",
    [
        dict(root_shape=0.0),
        dict(shape_per_parent=-1.0),
        dict(nonroot_shape=0.0),
        dict(ig_scale=0.0),
        dict(eta=0.0),
        dict(weight_mean=-0.1),
    ],
)
def test_validation(kw):
    args = dict(
        root_shape=2.5,
        shape_per_parent=0.5,
        nonroot_shape=None,
        ig_scale=0.5,
        eta=1.0,
        weight_mean=0.0,
    )
    with pytest.raises(ValueError):
        LinearGaussianPrior(**{**args, **kw})


def test_is_linear_gaussian():
    for k in (
        "linearnonidentifiable",
        "linear_nig_legacy",
        "linear_nig",
        "linear_iw",
    ):
        assert is_linear_gaussian(k) and k in REGISTRY
    assert not is_linear_gaussian("linearfixedstd")


def test_from_config_mapping():
    legacy = MixedRegimeConfig(
        **BASE,
        function_generator="linear_nig_legacy",
        weight_scale=4.0,
        weight_mean=0.5,
    )
    assert LinearGaussianPrior.from_config(legacy) == LinearGaussianPrior.legacy(
        4.0, 0.5
    )
    iw = MixedRegimeConfig(**BASE, function_generator="linear_iw", alpha0=1.5)
    assert LinearGaussianPrior.from_config(iw) == LinearGaussianPrior.iw(1.5)
    assert LinearGaussianPrior.from_config(
        MixedRegimeConfig(**BASE, function_generator="linear_iw")
    ) == LinearGaussianPrior.iw(2.5)
    nig = MixedRegimeConfig(
        **BASE, function_generator="linear_nig", alpha0=1.5, weight_scale=9.0
    )
    p = LinearGaussianPrior.from_config(nig)
    assert p.root_shape == 1.5 and p.shape_per_parent == 0.5 and p.eta == 9.0
    assert not p.is_iw_compatible


def test_linear_iw_rejects_non_iw_weight_prior():
    with pytest.raises(ValueError):
        MixedRegimeConfig(
            **BASE, function_generator="linear_iw", weight_scale=9.0
        )
    with pytest.raises(ValueError):
        MixedRegimeConfig(**BASE, function_generator="linear_iw", weight_mean=0.5)


def test_alpha0_only_for_new_families():
    with pytest.raises(ValueError):
        MixedRegimeConfig(
            **BASE, function_generator="linear_nig_legacy", alpha0=1.5
        )


def test_alias_warns_once_and_maps_to_legacy():
    assert LinearNonIdentifiable is LinearNIGLegacy
    lgp._ALIAS_WARNED = False
    with pytest.warns(DeprecationWarning, match="legacy NIG prior"):
        gen = REGISTRY["linearnonidentifiable"](
            num_variables=3, intervention_range_multiplier=1
        )
    assert isinstance(
        gen.generate_functions(np.zeros((3, 3)))[0], LinearNIGLegacy
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # second use must not warn
        REGISTRY["linearnonidentifiable"](
            num_variables=3, intervention_range_multiplier=1
        )


def test_linear_nig_draws_one_variance_with_node_shape():
    np.random.seed(0)
    prior = LinearGaussianPrior.iw(2.5)
    root, node = LinearNIG(0, prior), LinearNIG(2, prior)
    assert root.var.shape == (1,) and node.var.shape == (1,)
    out = node(np.ones((5, 2)))
    assert out.shape == (5, 1) and node.weight.shape == (2, 1)
    assert node.eta == 1.0


def test_task_meta_carries_prior():
    from CITNP.datasets.mixed_regime_generator import (
        SOURCE_ADHOC,
        generate_task,
        make_key,
    )

    cfg = MixedRegimeConfig(**BASE, function_generator="linear_iw", alpha0=1.5)
    task = generate_task(cfg, make_key(SOURCE_ADHOC, 0, 0, 0), num_variables=3)
    assert task.meta["is_iw_compatible"] is True
    assert task.meta["prior"]["root_shape"] == 1.5
    cfg = MixedRegimeConfig(**BASE, function_generator="linear_nig_legacy")
    task = generate_task(cfg, make_key(SOURCE_ADHOC, 0, 0, 0), num_variables=3)
    assert task.meta["is_iw_compatible"] is False


def test_oracle_prior_argument_matches_legacy_kwargs():
    from CITNP.evaluation.oracle import LinGaussOracle, node_posterior

    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 3))
    mask = np.zeros((30, 3), dtype=bool)
    pa = np.array([0, 1])
    old = node_posterior(x, mask, 2, pa, eta=4.0, weight_mean=0.5).log_ml
    new = node_posterior(
        x, mask, 2, pa, prior=LinearGaussianPrior.legacy(4.0, 0.5)
    ).log_ml
    assert old == new
    # IW differs from legacy only for nodes with >= 2 parents
    iw = LinearGaussianPrior.iw(2.5)
    one = np.array([0])
    assert (
        node_posterior(x, mask, 2, one, prior=iw).log_ml
        == node_posterior(x, mask, 2, one).log_ml
    )
    assert (
        node_posterior(x, mask, 2, pa, prior=iw).log_ml
        != node_posterior(x, mask, 2, pa).log_ml
    )
    assert LinGaussOracle(3, prior=iw).prior is iw
