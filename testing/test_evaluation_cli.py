"""Tests for the evaluation CLI's plotting commands and the deprecated module shims."""

import importlib
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import GOLDEN

KSHOT_FIGURES = ["main", "by_query", "effect", "flag", "varratio", "qtype", "by_t"]
SHIMS = [
    "eval_d3_kshot",
    "eval_d3_oracle",
    "oracle_bank",
    "oracle_lingauss",
    "plot_kshot_extra",
    "plot_monitoring",
]


@pytest.mark.skipif(not (GOLDEN / "kshot.csv").exists(), reason="golden files not recorded")
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
        old = importlib.reload(importlib.import_module("CITNP.evaluation.oracle_lingauss"))
    assert old.LinGaussOracle is oracle.LinGaussOracle
    assert old.make_context is tasks.make_context
