"""Tests for CITNP/prior_diagnostics (and the generator options it uses)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from CITNP.datasets.mixed_regime_generator import (  # noqa: E402
    SOURCE_ADHOC,
    MixedRegimeConfig,
    generate_task,
    make_key,
)
from CITNP.prior_diagnostics import __main__ as cli  # noqa: E402
from CITNP.prior_diagnostics import examples as ppe  # noqa: E402
from CITNP.prior_diagnostics import obs_vs_int as ovi  # noqa: E402
from CITNP.prior_diagnostics import population as pps  # noqa: E402
from CITNP.prior_diagnostics import stats as ps  # noqa: E402

RAW = {
    "function_generator": "linearnonidentifiable",
    "num_variables": [3],
    "graph_type": ["ER"],
    "graph_degrees": {3: [3]},  # full DAG on 3 nodes
    "n_ctx_range": [64, 128],
    "n_qry": 16,
}
CFG = MixedRegimeConfig(**RAW)


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(RAW))
    return str(p)


# ----------------------------------------------------------------- generator
def test_defaults_unchanged():
    key = make_key(SOURCE_ADHOC, 0, 0, 3)
    a, b = generate_task(CFG, key), generate_task(CFG, key, permute=True)
    assert np.array_equal(a.ctx_x, b.ctx_x) and np.array_equal(a.graph, b.graph)


# --------------------------------------------------------------------- stats
def test_path_lengths_and_relations():
    # 0 -> 1 -> 2, 3 -> 2, 3 -> 5, 4 isolated
    G = np.zeros((6, 6), int)
    G[0, 1] = G[1, 2] = G[3, 2] = G[3, 5] = 1
    dist = ps.path_lengths(G)
    assert dist[0, 2] == 2 and np.isinf(dist[2, 0])
    assert ps.relation(dist, 0, 1) == "direct"
    assert ps.relation(dist, 0, 2) == "downstream_2"
    assert ps.relation(dist, 2, 0) == "upstream"
    assert ps.relation(dist, 2, 5) == "confounded"
    assert ps.relation(dist, 0, 4) == "independent"


def test_edge_strengths_recovers_linear_coefficient():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=20000)
    x1 = 0.8 * x0 + 0.6 * rng.normal(size=20000)
    W = ps.edge_strengths(np.c_[x0, x1], np.array([[0, 1], [0, 0]]))
    assert abs(W[0, 1] - 0.8) < 0.03 and W[1, 0] == 0


def test_pair_weights_sum_to_one_per_task():
    task = generate_task(CFG, make_key(SOURCE_ADHOC, 0, 0, 1))
    for recs in (ps.query_pair_records(task), ps.context_pair_records(task)):
        if recs:
            assert abs(sum(r["weight"] for r in recs) - 1) < 1e-9


# ------------------------------------------------------------------ examples
def test_regime_label():
    assert ppe.regime_label(()) == "obs"
    assert ppe.regime_label((2,)) == "do(X2)"


def test_plot_task_panels():
    task = ppe.sample_example_tasks(CFG, 1)[0]
    fig = ppe.plot_task(task, max_pairs=6, task_id=0)
    assert len([a for a in fig.axes if a.get_xlabel()]) == 3  # D=3 -> 3 pairs
    plt.close(fig)


# ---------------------------------------------------------------- obs_vs_int
def test_probe_fixed_values_and_quantiles():
    base = generate_task(CFG, make_key(SOURCE_ADHOC, 0, 0, 0))
    probe, t = ovi.probe_task(CFG, 0, do_values=(-1.5, 2.0), n_obs=300, n_per_do=200)
    assert np.array_equal(probe.graph, base.graph)  # labels kept
    for r, x in [(1, -1.5), (2, 2.0)]:
        rows = probe.ctx_regime == r
        assert rows.sum() == 200 and np.allclose(probe.ctx_x[rows, t], x)
    q, t = ovi.probe_task(CFG, 0, do_quantiles=(0.5,), n_obs=300, n_per_do=200)
    med = np.median(q.ctx_x[q.ctx_regime == 0, t])
    assert np.isclose(q.ctx_x[q.ctx_regime == 1, t][0], med, atol=1e-6)


# ----------------------------------------------------------------------- CLI
def test_cli_examples(tmp_path, cfg_file):
    out = tmp_path / "ex.pdf"
    cli.main(["examples", "--config", cfg_file, "--num_tasks", "2", "--out", str(out), "--png"])
    assert out.exists() and (tmp_path / "ex_001.png").exists()


def test_cli_obs_vs_int(tmp_path, cfg_file):
    out = tmp_path / "ovi.pdf"
    cli.main(["obs-vs-int", "--config", cfg_file, "--num_tasks", "1", "--n_obs", "300",
              "--n_per_do", "200", "--out", str(out), "--png"])  # fmt: skip
    assert out.exists() and (tmp_path / "ovi_000.png").exists()


def test_cli_stats(tmp_path, cfg_file):
    out = tmp_path / "stats.pdf"
    cli.main(["stats", "--config", cfg_file, "--num_tasks", "20",
              "--n_probe_tasks", "5", "--n_rows", "200", "--out", str(out)])  # fmt: skip
    assert out.exists() and (tmp_path / "stats_effects.csv").exists()


def test_cli_summary(tmp_path, cfg_file):
    out = tmp_path / "prior.png"
    cli.main(["summary", "--config", cfg_file, "--num_tasks", "20", "--out", str(out)])
    assert out.exists()


def test_cli_unknown_command():
    with pytest.raises(SystemExit) as e:
        cli.main(["nope"])
    assert e.value.code == 2
    plt.close("all")
