"""Unit tests for CITNP/evaluation: tasks, metrics, oracle, experiments and
model_api.load_model (CPU, random tiny model; no golden files needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import REPO, TINY_CFG, build_tiny_model

from CITNP.evaluation import metrics, tasks
from CITNP.evaluation.oracle import LinearSCM, LinGaussOracle, all_dags


@pytest.fixture(scope="module")
def scm() -> LinearSCM:
    return tasks.confounder_scm()


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
def test_queries_and_names():
    assert tasks.QUERIES == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    assert tasks.qname(0, 2) == tasks.MAIN_QUERY == "outcome|do(treatment)"


@pytest.mark.parametrize(
    "nodes, T, Y, expected",
    [
        (set(), 0, 2, "unidentified"),
        ({0}, 0, 2, "seen"),
        ({0}, 1, 2, "unseen_identified"),
        ({1}, 1, 0, "seen"),
        ({1}, 0, 1, "unseen_identified"),
        ({1}, 0, 2, "unidentified"),
        ({2}, 2, 0, "seen"),
        ({2}, 0, 2, "unidentified"),
        ({1, 2}, 0, 2, "unseen_identified"),
    ],
)
def test_query_type(nodes, T, Y, expected):
    assert tasks.query_type(frozenset(nodes), T, Y) == expected


def test_standardise_gives_unit_variances(scm):
    np.testing.assert_allclose(np.diag(metrics.obs_cov(scm)), 1.0, rtol=1e-10)
    rnd = tasks.random_confounder_scm(np.random.default_rng(3))
    np.testing.assert_allclose(np.diag(metrics.obs_cov(rnd)), 1.0, rtol=1e-10)
    assert (rnd.adj == scm.adj).all()  # same graph, other weights


@pytest.mark.parametrize("num_nodes", [3, 4])
def test_context_designs_keep_observational_rows(num_nodes):
    designs = tasks.context_designs(num_nodes=num_nodes)
    assert len(designs) == 1 + num_nodes * 5
    for _, regimes in designs:
        assert regimes[0] == (None, 128)  # normalisation needs obs rows


def test_make_context(scm):
    x, m = tasks.make_context(
        scm, [(None, 40), (2, 8), (0, 0)], 1.5, np.random.default_rng(0)
    )
    assert x.shape == m.shape == (48, 3)
    assert m.sum() == 8 and m[:, 2].sum() == 8
    x2, m2 = tasks.make_context(
        scm, [(None, 40), (2, 8)], 1.5, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(x, x2)  # deterministic given the generator


def test_make_pools(scm):
    pools = tasks.make_pools(scm, 7, 1.5, np.random.default_rng(1))
    assert set(pools) == set(tasks.DESIGNS)
    for design, (x, m, seq) in pools.items():
        assert x.shape == m.shape == (7, 3) and seq.shape == (7,)
        np.testing.assert_array_equal(m.sum(1), 1)  # one intervened node per row
        np.testing.assert_array_equal(np.argmax(m, 1), seq)
        if design in tasks.FIXED_NODE:
            assert (seq == tasks.FIXED_NODE[design]).all()
    again = tasks.make_pools(scm, 7, 1.5, np.random.default_rng(1))
    np.testing.assert_array_equal(pools["random"][0], again["random"][0])


@pytest.mark.parametrize(
    "n, bucket", [(0, "nT=0"), (1, "nT=1-10"), (10, "nT=1-10"), (11, "nT=11+")]
)
def test_n_int_bucket(n, bucket):
    assert tasks.n_int_bucket(n) == bucket


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_mix_logpdf_single_component_is_normal():
    mu = torch.tensor([[0.5], [-1.0]], dtype=torch.float64)
    sd = torch.tensor([[0.3], [2.0]], dtype=torch.float64)
    lw = torch.zeros(2, 1, dtype=torch.float64)
    y = torch.tensor([[0.0, 1.0], [2.0, -3.0]], dtype=torch.float64)
    expected = torch.distributions.Normal(mu, sd).log_prob(y)
    torch.testing.assert_close(metrics.mix_logpdf(y, mu, sd, lw), expected)


def test_kl_mc_is_zero_for_identical_densities():
    lp = torch.randn(4, 9, dtype=torch.float64)
    np.testing.assert_array_equal(metrics.kl_mc(lp, lp), np.zeros(4))


def test_moments_and_implied_slope():
    grid = np.linspace(-2, 2, 5)
    mu = np.stack([1.5 * grid + 0.2, 1.5 * grid + 0.2], -1)
    sd = np.full_like(mu, 0.5)
    logw = np.log(np.full_like(mu, 0.5))
    mean, var = metrics.mixture_moments(mu, sd, logw)
    np.testing.assert_allclose(mean, 1.5 * grid + 0.2)
    np.testing.assert_allclose(var, 0.25, atol=1e-12)
    np.testing.assert_allclose(
        metrics.implied_slope(grid, mean), (1.5, 0.2), atol=1e-12
    )
    assert metrics.variance_ratio(var, var) == pytest.approx(1.0)


def test_oracle_moments_single_component():
    w, slope, var = np.array([1.0]), np.array([-0.4]), np.array([0.3])
    grid = np.linspace(-1, 1, 3)
    s_bar, s_sd, mean_o, var_o = metrics.oracle_moments(w, slope, var, grid)
    assert s_bar == pytest.approx(-0.4) and s_sd == pytest.approx(0.0)
    np.testing.assert_allclose(mean_o, -0.4 * grid)
    np.testing.assert_allclose(var_o, 0.3)


def test_naive_slope_is_confounded(scm):
    S = metrics.obs_cov(scm)
    b_true, _ = scm.do_dist(0, 2)
    assert b_true < 0 < metrics.naive_slope(S, 0, 2)  # the sign flip


# --------------------------------------------------------------------------- #
# oracle
# --------------------------------------------------------------------------- #
def test_oracle_posterior(scm):
    assert len(all_dags(3)) == 25
    x, m = tasks.make_context(
        scm, [(None, 64), (0, 16)], 1.5, np.random.default_rng(0)
    )
    orc = LinGaussOracle(3).fit(x, m)
    assert orc.graph_post.sum() == pytest.approx(1.0)
    assert 0.0 <= orc.prob_ancestor(0, 2) <= 1.0
    t = np.linspace(-1, 1, 5)
    lp = orc.logpdf(0, 2, t, np.zeros(5), n_mc=50, seed=0)
    assert lp.shape == (5,) and np.isfinite(lp).all()


# --------------------------------------------------------------------------- #
# experiments.run_bank_gap and model_api.load_model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    return build_tiny_model().eval()


def _bank(scm) -> list[dict]:
    rng = np.random.default_rng(0)
    items = []
    for n_T, label in ((0, "nT=0"), (5, "nT=1-10")):
        x, m = tasks.make_context(scm, [(None, 40), (0, n_T)], 1.5, rng)
        t = rng.normal(0, 1.5, 6)
        y = rng.normal(0, 1, 6)
        items.append(dict(
            ctx_x=x.astype(np.float32), ctx_int=m, T=0, Y=2,
            t=t.astype(np.float32), y=y.astype(np.float32),
            oracle_logp=np.zeros(6), n_int_T=n_T, labels=["all", label],
        ))  # fmt: skip
    return items


def test_run_scm_suite_direct(model):
    """Calls run_scm_suite directly (its model NLL goes through
    predict_mixture -> mixture_logpdf), independent of the golden files."""
    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.evaluation.experiments import run_scm_suite
    from CITNP.evaluation.model_api import mixture_logpdf, predict_mixture

    data_cfg = load_config(str(REPO / "CITNP/experiments/configs/oracle_d3.yaml"))
    df = run_scm_suite(
        model, data_cfg, device="cpu", n_seeds=1, n_query=8, n_mc=10
    )
    suite, designs = tasks.scm_suite(), tasks.context_designs()
    assert len(df) == len(suite) * len(designs)
    assert list(df.scm.unique()) == [name for name, *_ in suite]
    for col in ("truth", "oracle", "model", "model_noflag", "model_minus_oracle"):
        assert np.isfinite(df[col]).all(), col

    # The "model" column of the first row by hand: same context and queries as
    # run_scm_suite (seed 0), same call into model_api.
    name, scm, (T, Y), _ = suite[0]
    _, regimes = designs[0]
    rng = np.random.default_rng(7)
    x, m = tasks.make_context(scm, regimes, 1.5, rng)
    t = rng.normal(0, 1.5, 8)
    y = scm.sample(8, T, t, rng)[0][:, Y]
    mu, sd, logw = predict_mixture(model, x, m, T, Y, t, device="cpu")
    expected = -mixture_logpdf(mu, sd, logw, y)[0].mean().item()
    assert df.loc[0, "model"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("flag_ablation", [True, False])
def test_run_bank_gap_columns(model, scm, flag_ablation):
    from CITNP.evaluation.experiments import run_bank_gap

    row = run_bank_gap(
        model, _bank(scm), device="cpu", flag_ablation=flag_ablation
    )
    expected = [
        "oracle_gap/all", "oracle_gap/nT=0", "oracle_gap/nT=1-10",
        "calib/cover50", "calib/cover90", "mog/effective_components",
    ] + (["flag/logp_gain"] if flag_ablation else [])  # fmt: skip
    assert list(row) == expected
    assert all(np.isfinite(v) for v in row.values())
    assert 0.0 <= row["calib/cover50"] <= row["calib/cover90"] <= 1.0


def test_load_model_class_from_yaml(tmp_path):
    from CITNP.evaluation.model_api import load_model

    ckpt = tmp_path / "tiny.pt"
    torch.save(build_tiny_model().state_dict(), ckpt)
    cfg_text = (REPO / TINY_CFG).read_text()
    default = load_model(str(ckpt), str(REPO / TINY_CFG), "cpu")
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(cfg_text + "\nmodel_class: MixedContextMoGModel\n")
    model = load_model(str(ckpt), str(explicit), "cpu")
    assert type(model) is type(default) and not model.training
    for a, b in zip(model.state_dict().values(), default.state_dict().values()):
        torch.testing.assert_close(a, b)
    bad = tmp_path / "bad.yaml"
    bad.write_text(cfg_text + "\nmodel_class: NoSuchModel\n")
    with pytest.raises(AttributeError):
        load_model(str(ckpt), str(bad), "cpu")
