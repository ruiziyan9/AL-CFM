"""Tests for the evaluation CLI's plotting commands and the deprecated module shims."""

import importlib
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import DATA_CFG, GOLDEN, REPO, TINY_CFG

KSHOT_FIGURES = [
    "main",
    "by_query",
    "effect",
    "flag",
    "varratio",
    "qtype",
    "by_t",
]
SHIMS = [
    "eval_d3_kshot",
    "eval_d3_oracle",
    "oracle_bank",
    "oracle_lingauss",
    "plot_kshot_extra",
    "plot_monitoring",
]


@pytest.mark.skipif(
    not (GOLDEN / "kshot.csv").exists(), reason="golden files not recorded"
)
def test_plot_kshot_from_csv(tmp_path):
    from CITNP.evaluation.__main__ import main

    for name in ("kshot.csv", "kshot_by_t.csv"):
        shutil.copy(GOLDEN / name, tmp_path / name)
    main(["plot-kshot", str(tmp_path / "kshot.csv")])
    for fig in KSHOT_FIGURES:
        assert (tmp_path / f"kshot_{fig}.png").stat().st_size > 0, fig


def test_plot_monitoring(tmp_path):
    from CITNP.evaluation.__main__ import main

    run = tmp_path / "run_a"
    run.mkdir()
    (run / "oracle_gap.csv").write_text(
        "step,oracle_gap/all,calib/cover90,flag/logp_gain\n0,0.5,0.8,0.1\n10,0.3,0.9,0.2\n"
    )
    (run / "train_stats.csv").write_text("step,train/loss_mean\n10,1.0\n20,0.8\n")
    out = tmp_path / "monitoring.png"
    main(["plot-monitoring", str(run), "--out", str(out)])
    assert out.stat().st_size > 0


def test_plots_does_not_import_torch():
    import subprocess

    code = "import sys, CITNP.evaluation.plots; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", SHIMS)
def test_shim_warns_and_has_main(name):
    mod = importlib.import_module(f"CITNP.evaluation.{name}")
    with pytest.warns(DeprecationWarning):
        mod = importlib.reload(mod)
    if name != "oracle_lingauss":
        assert callable(mod.main)


def test_oracle_lingauss_reexports():
    from CITNP.evaluation import oracle, tasks

    with pytest.warns(DeprecationWarning):
        old = importlib.reload(
            importlib.import_module("CITNP.evaluation.oracle_lingauss")
        )
    assert old.LinGaussOracle is oracle.LinGaussOracle
    assert old.make_context is tasks.make_context


# --------------------------------------------------------------------------- #
# kshot-prior
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (GOLDEN / "tiny_model.pt").exists(), reason="golden files not recorded"
)
def test_kshot_prior_smoke(tmp_path):
    import json

    import pandas as pd

    from CITNP.evaluation.__main__ import main

    out = tmp_path / "kp.csv"
    main([
        "kshot-prior", "--ckpt", str(GOLDEN / "tiny_model.pt"),
        "--model_config", str(REPO / TINY_CFG),
        "--data_config", str(REPO / DATA_CFG),
        "--n_tasks", "2", "--k_list", "0,1", "--n_mc", "50", "--n_y", "16",
        "--device", "cpu", "--out", str(out),
    ])  # fmt: skip
    df = pd.read_csv(out)
    assert (tmp_path / "kp_by_t.csv").stat().st_size > 0
    assert set(df["design"]) == {"obs", "random", "do(T)", "do(Y)", "do(other)"}
    assert set(df["k"]) == {0, 1} and set(df["task"]) == {0, 1}
    assert df["kl_oracle_model"].notna().all()
    meta = json.loads((tmp_path / "kp_meta.json").read_text())
    assert len(meta["data_config"]["sha256"]) == 64
    assert meta["num_variables"] == [3] and meta["k_list"] == [0, 1]
    assert sum(v["n_tasks"] for v in meta["do_dist_summary"].values()) == 2


def test_kshot_prior_rejects_foreign_num_variables(tmp_path):
    from CITNP.evaluation.__main__ import main

    with pytest.raises(SystemExit):
        main([
            "kshot-prior", "--ckpt", "x", "--model_config", "y",
            "--data_config", str(REPO / DATA_CFG), "--num_variables", "5",
            "--out", str(tmp_path / "kp.csv"),
        ])  # fmt: skip


def test_plot_kshot_prior_refuses_mixed_data_configs(tmp_path):
    import json

    from CITNP.evaluation.__main__ import main

    paths = []
    for i, h in enumerate(["a" * 64, "b" * 64]):
        csv = tmp_path / f"p_part{i}.csv"
        csv.write_text("D\n3\n")
        (tmp_path / f"p_part{i}_meta.json").write_text(
            json.dumps({"data_config": {"sha256": h}})
        )
        paths.append(str(csv))
    with pytest.raises(SystemExit):
        main(["plot-kshot-prior", *paths])


def test_plot_kshot_prior_writes_figures(tmp_path):
    import itertools
    import json

    import numpy as np
    import pandas as pd

    from CITNP.evaluation.__main__ import main

    rng = np.random.default_rng(0)
    rows = []
    for task, (T, Y) in itertools.product(range(3), [(0, 1), (1, 2)]):
        for design, k in [("obs", 0)] + [
            (d, k) for d in ("random", "do(T)", "do(Y)", "do(other)")
            for k in (1, 3)
        ]:  # fmt: skip
            om = float(rng.uniform(0.0, 0.2))
            to = float(rng.uniform(0.1, 0.5))
            rows.append({
                "D": 3, "task": task, "T": T, "Y": Y,
                "relation": "reverse" if task else "unrelated",
                "design": design, "k": k, "kl_truth_oracle": to,
                "kl_truth_model": to + om, "kl_oracle_model": om,
                "excess_kl_truth": om,
            })  # fmt: skip
    csv = tmp_path / "kp_part0.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    (tmp_path / "kp_part0_meta.json").write_text(
        json.dumps({"data_config": {"sha256": "a" * 64}})
    )
    main(["plot-kshot-prior", str(csv)])
    for s in ("main", "by_relation"):
        assert (tmp_path / f"kp_D3_{s}.png").stat().st_size > 0
