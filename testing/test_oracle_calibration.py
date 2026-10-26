"""Fast checks of the oracle calibration (S2.4) and resample diagnostic (S2.3)."""

import numpy as np
import pytest

from CITNP.datasets.mixed_regime_generator import MixedRegimeConfig
from CITNP.evaluation.oracle import all_dags, dag_key
from CITNP.evaluation.oracle_calibration import (
    COLUMNS,
    calibrate,
    reliability_table,
    summarise,
)
from CITNP.evaluation.tasks import BUCKETS
from CITNP.prior_diagnostics.resample import resample_rate

CFG = dict(
    function_generator="linear_iw",
    alpha0=1.5,
    num_variables=[3],
    graph_type=["ER"],
    graph_degrees={3: [1, 2, 3]},
    normalise=False,
    k_max=2,
    n_ctx_range=(30, 40),
    n_qry=6,
    qry_mode="grouped",
    n_obs_min=10,
)


@pytest.fixture(scope="module")
def calib():
    cfg = MixedRegimeConfig(**CFG)
    # cheap graph prior: uniform over the 25 DAGs
    gp = {dag_key(g): -np.log(25) for g in all_dags(3)}
    return calibrate(cfg, 5, n_mc=50, compare_mc=True, log_graph_prior=gp)


def test_columns_and_ranges(calib):
    df, mec, info = calib
    assert list(df.columns) == COLUMNS
    assert len(df) > 0
    assert df.pit.between(0, 1).all() and df.p_true_mec.between(0, 1).all()
    assert set(df.nT_bucket) <= set(BUCKETS)
    assert mec.p.between(0, 1).all()
    assert mec.groupby("task").p.sum().round(6).eq(1).all()
    assert mec.groupby("task").is_true.sum().eq(1).all()
    assert info["mc_abs_dnll"] >= 0


def test_summary_runs(calib):
    df, mec, info = calib
    out = summarise(df, mec, info)
    assert "reliability_max_gap" in out
    assert reliability_table(mec).n.sum() == len(mec)


def test_resample_rate_counts_rejections():
    cfg = MixedRegimeConfig(
        **{**CFG, "max_abs_value": 1e-3, "max_resample_attempts": 2}
    )
    r = resample_rate(cfg, 3)
    assert r["n_failed"] == 3 and r["n_rejected"] == 9 and r["rate"] == 1.0
    r = resample_rate(MixedRegimeConfig(**CFG), 3)
    assert r["n_failed"] == 0 and 0 <= r["rate"] < 1
