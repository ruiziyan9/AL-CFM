"""Tests for prior_diagnostics.linear_gaussian, the question classes in
prior_diagnostics.stats, and prior_diagnostics.obs_vs_do."""

import matplotlib

matplotlib.use("Agg")

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
from CITNP.prior_diagnostics import obs_vs_do as ovd  # noqa: E402
from CITNP.prior_diagnostics import stats as ps  # noqa: E402
from CITNP.prior_diagnostics.linear_gaussian import scm_from_debug  # noqa: E402
from CITNP.prior_diagnostics.obs_vs_int import make_probe_design  # noqa: E402

RAW = {
    "function_generator": "linearnonidentifiable",
    "num_variables": [3, 4],
    "graph_type": ["ER"],
    "graph_degrees": {3: [2, 3], 4: [2, 3, 4]},
    "n_ctx_range": [64, 128],
    "n_qry": 48,
    "qry_mode": "grouped",
    "qry_groups": [1, 2, 3],
}


def test_debug_does_not_change_task():
    cfg = MixedRegimeConfig(**RAW)
    key = make_key(SOURCE_ADHOC, 0, 0, 4)
    a = generate_task(cfg, key)
    b, debug = generate_task(cfg, key, return_debug=True)
    assert np.array_equal(a.ctx_x, b.ctx_x) and a.meta.keys() == b.meta.keys()
    assert "_mechanisms" not in b.meta and len(debug["mechanisms"]) == len(a.graph)


@pytest.mark.parametrize("normalise", [True, False])
def test_exact_moments_match_simulation(normalise):
    cfg = MixedRegimeConfig(**{**RAW, "normalise": normalise, "qry_mode": "per_row"})
    key = make_key(SOURCE_ADHOC, 0, 0, 7)
    base = generate_task(cfg, key, num_variables=3)
    t, x, n = 0, 0.7, 40000
    design = make_probe_design(3, t, [x], n_obs=n, n_per_do=n)
    task, debug = generate_task(cfg, key, graph=base.graph, design=design,
                                permute=False, return_debug=True)  # fmt: skip
    scm = scm_from_debug(debug)
    mu, S = scm.moments()
    sd = np.sqrt(np.diag(S))
    obs = task.ctx_x[task.ctx_regime == 0].astype(float)
    assert np.allclose(obs.mean(0), mu, atol=0.05 * sd.max())
    assert np.allclose(np.cov(obs.T), S, atol=0.05 * sd.max() ** 2)
    m_do, _ = scm.moments({t: x})
    do_rows = task.ctx_x[task.ctx_regime == 1].astype(float)
    assert np.allclose(do_rows.mean(0), m_do, atol=0.05 * sd.max())


def test_question_classes_and_roles():
    G = np.zeros((4, 4), int)
    G[2, 0] = G[2, 1] = G[0, 1] = 1  # 2 confounds 0 -> 1
    G[1, 3] = 1  # 3 is a child of Y=1
    assert ps.question_class(G, 0, 1) == "confounded effect"
    assert ps.question_class(G, 1, 0) == "reverse"
    assert ps.question_class(G, 0, 3) == "confounded effect"  # 0 <- 2 -> 1 -> 3
    H = np.zeros((3, 3), int)
    H[0, 2] = H[2, 1] = 1  # 0 -> 2 -> 1
    assert ps.question_class(H, 0, 1) == "unconfounded effect"
    C = np.zeros((3, 3), int)
    C[0, 2] = C[1, 2] = 1  # collider at 2
    assert ps.question_class(C, 0, 1) == "unrelated"
    P = np.zeros((3, 3), int)
    P[2, 0] = P[2, 1] = 1
    assert ps.question_class(P, 0, 1) == "pure confounding"
    # back-door source whose only path to Y goes through T is not a confounder
    Q = np.zeros((3, 3), int)
    Q[2, 0] = Q[0, 1] = 1  # 2 -> 0 -> 1
    assert ps.question_class(Q, 0, 1) == "unconfounded effect"
    assert ps.context_role(Q, 0, 1, 2) == "do(other)"
    assert ps.context_role(G, 0, 1, 2) == "do(confounder)"
    assert ps.context_role(H, 0, 1, 2) == "do(mediator)"
    assert ps.context_role(G, 0, 1, 3) == "do(other)"
    assert ps.context_role(G, 0, 1, None) == "obs"


def test_identical_classes_have_zero_gap():
    recs, _ = ovd.collect_obs_vs_do(MixedRegimeConfig(**RAW), 30)
    for r in recs:
        if r["cls"] not in ps.DIFFER_CLASSES:
            assert r["w2_sd"] < 1e-6 and r["evidence_nats"] < 1e-6


def test_cli_obs_vs_do(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(RAW))
    out = tmp_path / "ovd.pdf"
    cli.main(["obs-vs-do", "--config", str(p), "--num_tasks", "40",
              "--num_variables", "3", "--out", str(out), "--png"])  # fmt: skip
    assert out.exists() and out.with_suffix(".csv").exists() and out.with_suffix(".png").exists()


def test_select_example_weighted_median():
    recs = [{"cls": "reverse", "w2_sd": g, "weight": w, "task": i}
            for i, (g, w) in enumerate([(0.1, 1), (0.9, 1), (0.5, 1), (2.0, 0.1)])]
    assert ovd.select_example(recs, "reverse")["w2_sd"] == 0.5
    assert ovd.select_example(recs, "reverse", "first")["task"] == 0
    assert ovd.select_example(recs, "pure confounding") is None
    with pytest.raises(ValueError):
        ovd.select_example(recs, "reverse", "mean")


def test_examples_are_class_medians():
    recs, ex = ovd.collect_obs_vs_do(MixedRegimeConfig(**RAW), 30)
    for cls, (task, scm, t, y, rec) in ex.items():
        assert rec is ovd.select_example(recs, cls)
        assert ps.question_class(task.graph, t, y) == cls  # regenerated task is the same task