"""kshot-example: one fixed model and one query over repeated context draws
(run_kshot_example), with injected predictors (CPU, no checkpoint)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from CITNP.evaluation import example_model as em
from CITNP.evaluation import tasks
from CITNP.evaluation.experiments import run_kshot_example

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kshot_prior import (  # noqa: E402
    CFG,
    CFG_FIVE_VARIABLES,
    SEED,
    normal_predictor,
    uniform_prior,
    untrained_model_predictor,
)

# A three-variable example (oracle-sized) and a five-variable one (above it).
GRAPH_D3 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
FIVE_EDGES = [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4), (0, 2)]


def _graph_d5():
    graph = np.zeros((5, 5), dtype=int)
    for source, target in FIVE_EDGES:
        graph[source, target] = 1
    return graph


ORACLE_COLUMNS = [
    "kl_truth_oracle", "kl_oracle_model", "excess_kl_truth", "slope_oracle",
    "slope_sd_oracle", "slope_err_oracle_truth", "var_ratio_model_oracle",
]  # fmt: skip
ORACLE_COLUMNS_BY_DO_VALUE = [
    "kl_truth_oracle", "kl_oracle_model", "mean_oracle", "sd_oracle",
]  # fmt: skip


def example_d3():
    return em.draw_example_model(CFG, GRAPH_D3, SEED)


def example_d5():
    return em.draw_example_model(CFG_FIVE_VARIABLES, _graph_d5(), SEED)


def run_example(model, predict, *, T=0, Y=2, repetitions=3, **kw):
    kw.setdefault("n_mc", 50)
    kw.setdefault("n_y", 32)
    return run_kshot_example(
        predict, model.scm, model.graph, model.do_dist,
        T=T, Y=Y, device="cpu", k_list=[0, 2], n_obs=30,
        repetitions=repetitions, base_seed=SEED, n_t=5, **kw,
    )  # fmt: skip


def make_truth_predictor(scm, T, Y):
    """A predictor that returns exactly the true do-distribution N(b t, v)."""
    b, v = scm.do_dist(T, Y)

    def predict(x, m, Ts, Ys, grid, zero_flags=False):
        n_t = len(grid)
        mu = torch.tensor(
            b * np.asarray(grid, dtype=float), dtype=torch.float64
        ).reshape(1, n_t, 1)
        sd = torch.full((1, n_t, 1), float(np.sqrt(v)), dtype=torch.float64)
        logw = torch.zeros((1, n_t, 1), dtype=torch.float64)
        return mu, sd, logw

    return predict


# --------------------------------------------------------------------------- #
# designs, layout
# --------------------------------------------------------------------------- #
def test_designs_are_named_by_what_they_add():
    assert tasks.example_design_name(0) == "do(X1)"
    assert tasks.example_design_name(2) == "do(X3)"
    assert tasks.example_design_name("random") == "random"
    assert tasks.example_design_name("observational") == "observational"
    assert tasks.example_design_names(5) == [
        "do(X1)", "do(X2)", "do(X3)", "do(X4)", "do(X5)",
        "random", "observational",
    ]


def test_every_design_appears_for_the_single_query():
    model = example_d3()
    df, df_t = run_example(
        model, normal_predictor, with_oracle=True,
        data_cfg=CFG, log_graph_prior=uniform_prior(3),
    )
    # obs at k = 0, and do(X1..X3), random, observational at k = 2
    assert set(df["design"]) == {"obs", *tasks.example_design_names(3)}
    assert set(df[df["design"] != "obs"]["k"]) == {2}
    assert (df["T"] == 0).all() and (df["Y"] == 2).all()
    assert (df["D"] == 3).all()
    assert df["graph"].eq("0->1,0->2,1->2").all()
    # the observational design has no intervened nodes
    obs_design = df[df["design"] == "observational"]
    assert (obs_design["intervened_nodes"] == "").all()
    assert {"rep", "k", "t"} <= set(df_t.columns)


# --------------------------------------------------------------------------- #
# model numbers do not depend on the oracle
# --------------------------------------------------------------------------- #
def test_model_columns_equal_with_and_without_oracle():
    model = example_d3()
    predict = untrained_model_predictor(3)
    on, on_t = run_example(
        model, predict, with_oracle=True, data_cfg=CFG,
        log_graph_prior=uniform_prior(3),
    )
    off, off_t = run_example(model, predict, with_oracle=False)
    for both, alone, oracle_cols in (
        (on, off, ORACLE_COLUMNS),
        (on_t, off_t, ORACLE_COLUMNS_BY_DO_VALUE),
    ):
        assert list(alone.columns) == list(both.columns)
        assert both[oracle_cols].notna().all().all()
        assert alone[oracle_cols].isna().all().all()
        pd.testing.assert_frame_equal(
            alone.drop(columns=oracle_cols),
            both.drop(columns=oracle_cols),
            check_exact=True,
        )


# --------------------------------------------------------------------------- #
# five variables, model only
# --------------------------------------------------------------------------- #
def test_model_only_five_variables(monkeypatch):
    model = example_d5()
    df, df_t = run_example(
        model, untrained_model_predictor(5), T=2, Y=4, with_oracle=False
    )
    assert set(df["design"]) == {"obs", *tasks.example_design_names(5)}
    model_cols = ["kl_truth_model", "slope_truth", "slope_model"]
    assert np.isfinite(df[model_cols]).all().all()
    assert df[ORACLE_COLUMNS].isna().all().all()
    assert df_t[ORACLE_COLUMNS_BY_DO_VALUE].isna().all().all()
    assert df["slope_truth"].nunique() == 1  # one fixed model


def test_large_d_with_oracle_points_to_model_only():
    model = example_d5()
    with pytest.raises(ValueError, match="with_oracle=False"):
        run_example(model, normal_predictor, T=2, Y=4,
                    with_oracle=True, data_cfg=CFG_FIVE_VARIABLES)  # fmt: skip


def test_bad_query_is_rejected():
    model = example_d3()
    with pytest.raises(ValueError, match="distinct"):
        run_example(model, normal_predictor, T=1, Y=1, with_oracle=False)


# --------------------------------------------------------------------------- #
# repetitions differ; reproducible from the seed
# --------------------------------------------------------------------------- #
def test_repetitions_differ_and_the_run_is_reproducible():
    model = example_d3()
    df, _ = run_example(
        model, untrained_model_predictor(3), repetitions=5, with_oracle=False
    )
    # different context draws give different scores at a fixed design and k
    at_k2 = df[(df["design"] == "do(X1)") & (df["k"] == 2)]
    assert at_k2["kl_truth_model"].nunique() == 5
    # same seed, same result
    again, _ = run_example(
        model, untrained_model_predictor(3), repetitions=5, with_oracle=False
    )
    pd.testing.assert_frame_equal(df, again, check_exact=True)


# --------------------------------------------------------------------------- #
# a truth-returning predictor recovers the true effect at zero KL
# --------------------------------------------------------------------------- #
def test_truth_predictor_has_zero_kl_and_the_true_effect():
    model = example_d3()
    T, Y = 0, 2
    truth_predict = make_truth_predictor(model.scm, T, Y)
    df, _ = run_example(
        model, truth_predict, T=T, Y=Y, repetitions=4, n_y=400,
        with_oracle=False,
    )
    true_effect = model.true_effect(T, Y)
    assert np.allclose(df["slope_truth"], true_effect)
    # the model equals the truth: KL ~ 0 and the estimated effect ~ true effect
    assert df["kl_truth_model"].abs().max() < 0.02
    assert np.allclose(df["slope_model"], true_effect, atol=1e-6)
