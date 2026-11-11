"""kshot-prior: tasks drawn from the training prior, nested k-shot contexts,
designs, and the identity E[excess] = E[KL(oracle||model)] with injected
predictors (CPU, no checkpoint)."""

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from CITNP.datasets.linear_gaussian_prior import LinearGaussianPrior
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_ADHOC,
    MixedRegimeConfig,
    do_dist_quantile,
    generate_task,
    make_key,
    sample_do_values,
)
from CITNP.evaluation import experiments, tasks
from CITNP.evaluation.experiments import _build_contexts, run_kshot_prior
from CITNP.evaluation.metrics import (
    mix_logpdf,
    oracle_mixture_on_grid,
    oracle_moments,
    oracle_weights,
)
from CITNP.evaluation.model_api import predict_mixture
from CITNP.evaluation.oracle import LinGaussOracle, all_dags, dag_key
from CITNP.prior_diagnostics.linear_gaussian import scm_from_debug

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import tiny_model_cfg

CFG = MixedRegimeConfig(
    function_generator="linear_iw",
    alpha0=1.5,
    num_variables=[3, 4],
    graph_type=["ER"],
    graph_degrees={3: [1, 2, 3], 4: [1, 2, 4]},
    normalise=False,
    k_max=2,
    n_ctx_range=(30, 40),
    n_qry=6,
    qry_mode="grouped",
    n_obs_min=10,
)
SEED = 9101
# Five variables: more than the exact oracle enumerates (max_oracle_nodes).
CFG_FIVE_VARIABLES = MixedRegimeConfig(
    function_generator="linear_iw",
    alpha0=1.5,
    num_variables=[5],
    graph_type=["ER"],
    graph_degrees={5: [3, 5]},
    normalise=False,
    k_max=2,
    n_ctx_range=(30, 40),
    n_qry=6,
    qry_mode="grouped",
    n_obs_min=10,
)
# Columns of the two result tables that need the oracle: missing values when
# run_kshot_prior runs with with_oracle=False.
ORACLE_COLUMNS = [
    "kl_truth_oracle",
    "kl_oracle_model",
    "excess_kl_truth",
    "slope_oracle",
    "var_ratio_model_oracle",
    "log_prior_true_graph",
    "true_graph_in_prior",
]
ORACLE_COLUMNS_BY_DO_VALUE = [
    "kl_truth_oracle",
    "kl_oracle_model",
    "mean_oracle",
    "sd_oracle",
]


def uniform_prior(D: int) -> dict:
    """Not the generator's graph distribution: only for tests that do not
    need the exact oracle of CFG."""
    dags = all_dags(D)
    return {dag_key(g): -np.log(len(dags)) for g in dags}


@functools.lru_cache(maxsize=None)
def empirical_prior(D: int) -> dict:
    """The generator's graph distribution, as the exact oracle needs it."""
    return tasks.empirical_graph_prior(CFG, num_nodes=D)


def normal_predictor(x, m, Ts, Ys, grid, zero_flags=False):
    """N(0, 1) for every query."""
    shape = (len(Ts), len(grid), 1)
    z = torch.zeros(shape, dtype=torch.float64)
    return z, torch.ones(shape, dtype=torch.float64), z.clone()


def make_oracle_predictor(D: int, log_graph_prior: dict, n_mc: int):
    """The exact oracle itself, as a model-shaped predictor."""
    prior = LinearGaussianPrior.from_config(CFG)

    def predict(x, m, Ts, Ys, grid, zero_flags=False):
        orc = LinGaussOracle(D, log_graph_prior=log_graph_prior, prior=prior)
        orc.fit(x, m)
        g = torch.tensor(grid, dtype=torch.float64)
        parts = []
        for T, Y in zip(Ts, Ys):
            lw, sl, var = orc.predictive_components(
                int(T), int(Y), n_mc=n_mc, min_graph_prob=1e-4, seed=0
            )
            lw, _, var = oracle_weights(lw, var)
            parts.append(oracle_mixture_on_grid(lw, sl, var, g))
        K = max(p[0].shape[1] for p in parts)

        def pad(a, fill):
            out = torch.full((len(grid), K), fill, dtype=torch.float64)
            out[:, : a.shape[1]] = a
            return out

        mu = torch.stack([pad(p[0], 0.0) for p in parts])
        sd = torch.stack([pad(p[1], 1.0) for p in parts])
        lw = torch.stack([pad(p[2], -1e30) for p in parts])
        return mu, sd, lw

    return predict


def untrained_model_predictor(D: int):
    """Predictor of a small randomly initialised model for D variables (the
    size-s model config, seed 0)."""
    from CITNP.models.mixed_context_model import MixedContextMoGModel
    from CITNP.utils.configs import CausalInfModelConfig

    m = tiny_model_cfg()
    cfg = CausalInfModelConfig(
        d_model=m["d_model"],
        emb_depth=m["emb_depth"],
        dim_feedforward=m["dim_feedforward"],
        nhead=m["nhead"],
        dropout=0.0,
        num_layers_encoder=m["num_layers_encoder"],
        num_nodes=D,
        sample_attn_mode=m["sample_attn_mode"],
        linear_attention=m.get("linear_attention", False),
        mean_loss_across_samples=False,
        device="cpu",
        dtype="float32",
        num_mixture_components=m["num_mixture_components"],
    )
    torch.manual_seed(0)
    model = MixedContextMoGModel(cfg).eval()
    return functools.partial(predict_mixture, model, device="cpu")


def run(D, n_tasks, k_list, predict, n_mc=50, **kw):
    """run_kshot_prior on CFG; the oracle uses `empirical_prior(D)` unless
    `log_graph_prior` is passed (None: run_kshot_prior's own default)."""
    if "log_graph_prior" not in kw:
        kw["log_graph_prior"] = empirical_prior(D)
    return run_kshot_prior(
        predict,
        CFG,
        D=D,
        device="cpu",
        k_list=k_list,
        n_tasks=n_tasks,
        base_seed=SEED,
        n_y=32,
        n_t=5,
        n_mc=n_mc,
        **kw,
    )


def task_means(df: pd.DataFrame, col: str, by=("design", "k")):
    """Mean over pairs within a task, then (mean, SE) over tasks."""
    per_task = df.groupby(["task", *by])[col].mean().unstack("task")
    return per_task.mean(axis=1), per_task.std(axis=1) / np.sqrt(
        per_task.shape[1]
    )


# --------------------------------------------------------------------------- #
# 1. prior_scm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("D", [3, 4])
def test_prior_scm_matches_generator(D):
    for i in range(4):
        scm, graph, do_dist, attempts = tasks.prior_scm(CFG, i, SEED, D)
        task, debug = generate_task(
            CFG,
            make_key(SOURCE_ADHOC, D, SEED, i),
            num_variables=D,
            return_debug=True,
        )
        assert np.array_equal(scm.adj, task.graph.astype(int))
        assert np.array_equal(graph, task.graph.astype(int))
        assert do_dist == dict(task.meta["do_dist"])
        assert attempts == task.meta["resample_attempts"]
        lg = scm_from_debug(debug)
        for T in range(D):
            for Y in range(D):
                if T == Y:
                    continue
                _, slope, var = lg.interventional_line(T, Y)
                b, v = scm.do_dist(T, Y)
                assert b == pytest.approx(slope, abs=1e-9)
                assert v == pytest.approx(var, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# 2. do-value sampling and quantiles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "do_dist",
    [{"kind": "normal", "scale": 1.7}, {"kind": "uniform", "width": 2.5}],
)
def test_sample_and_quantile_agree(do_dist):
    q = np.linspace(0.02, 0.98, 13)
    v = do_dist_quantile(do_dist, q)
    assert np.all(np.diff(v) > 0)
    s = sample_do_values(do_dist, 40_000, np.random.default_rng(0))
    emp = (s[:, None] <= v[None]).mean(0)
    assert np.abs(emp - q).max() < 0.01


def test_unknown_do_kind_raises():
    with pytest.raises(ValueError):
        do_dist_quantile({"kind": "cauchy"}, 0.5)
    with pytest.raises(ValueError):
        sample_do_values({"kind": "cauchy"}, 3, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# 3. contexts: nested in k, built once per context
# --------------------------------------------------------------------------- #
def test_contexts_nested_in_k():
    scm, _, do_dist, _ = tasks.prior_scm(CFG, 0, SEED, 3)
    rng = np.random.default_rng(0)
    ox, om = scm.sample(10, None, None, rng)
    pools = tasks.node_pools(scm, 5, do_dist, rng)
    keys = [0, 1, 2, "random"]
    ctx = _build_contexts(ox, om, pools, [1, 3], [SEED, 3, 0], designs=keys)
    assert ctx[0][0] == "obs" and ctx[0][1] == 0
    assert len(ctx) == 1 + 4 * 2
    by = {(d, k): x for d, k, x, _, _ in ctx}

    def rows(x):
        return {tuple(np.round(r, 9)) for r in x}

    for d in keys:
        assert len(by[(d, 1)]) == 11 and len(by[(d, 3)]) == 13
        assert rows(by[(d, 1)]) <= rows(by[(d, 3)])
        assert rows(ox) <= rows(by[(d, 1)])


def test_one_fit_per_context_shared_by_pairs():
    D, calls = 3, []

    def spy(x, m, Ts, Ys, grid, zero_flags=False):
        calls.append((x.copy(), len(Ts)))
        return normal_predictor(x, m, Ts, Ys, grid)

    # counts calls only: any graph prior will do
    df, _ = run(D, 2, [0, 2, 4], spy, log_graph_prior=uniform_prior(D))
    # per task: obs + (D + 1) pools x 2 positive k
    assert len(calls) == 2 * (1 + (D + 1) * 2)
    assert all(n == D * (D - 1) for _, n in calls)
    assert len(df) == 2 * 6 * (1 + 4 * 2)  # obs + 4 designs x 2 k


# --------------------------------------------------------------------------- #
# 4, 5. designs and context size
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("D", [3, 4])
def test_pair_design_pools_cover_each_node_once(D):
    for T in range(D):
        for Y in range(D):
            if T == Y:
                continue
            p = tasks.pair_design_pools(T, Y, D)
            assert set(p) == set(tasks.PRIOR_DESIGNS)
            assert p["random"] == "random"
            assert sorted([p["do(T)"], p["do(Y)"], *p["do(other)"]]) == list(
                range(D)
            )


def test_check_context_size():
    tasks.check_context_size(CFG, 30, 10)
    with pytest.raises(ValueError):
        tasks.check_context_size(CFG, 30, 11)
    assert tasks.default_n_obs(CFG) == 30


# --------------------------------------------------------------------------- #
# 6. identity with injected predictors
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_oracle_as_model_has_no_excess():
    D = 3
    # oracle vs itself: equal whatever the (shared) graph prior
    df, _ = run(
        D,
        40,
        [0, 3],
        make_oracle_predictor(D, uniform_prior(D), 200),
        n_mc=200,
        log_graph_prior=uniform_prior(D),
    )
    assert df["kl_oracle_model"].mean() < 0.05
    assert abs(df["excess_kl_truth"].mean()) < 0.05


@pytest.mark.slow
def test_normal_predictor_obeys_identity():
    df, _ = run(3, 100, [0, 3], normal_predictor, n_mc=200)
    ex, _ = task_means(df, "excess_kl_truth")
    assert (ex > 0).all()
    per_task = df.groupby("task")[["excess_kl_truth", "kl_oracle_model"]].mean()
    d = per_task["excess_kl_truth"] - per_task["kl_oracle_model"]
    se = d.std() / np.sqrt(len(d))
    assert abs(d.mean()) < 3 * se
    assert per_task["kl_oracle_model"].mean() > 0


# --------------------------------------------------------------------------- #
# 7. D = 4 smoke
# --------------------------------------------------------------------------- #
def test_d4_smoke():
    # smoke test: the uniform prior keeps it cheap (no D = 4 graph sampling)
    df, df_t = run(
        4, 2, [0, 2], normal_predictor, log_graph_prior=uniform_prior(4)
    )
    other = df[df["design"] == "do(other)"]
    assert len(other) > 0 and (other["n_other"] == 2).all()
    assert (df[df["design"] != "do(other)"]["n_other"] == 0).all()
    assert set(df["design"]) == {"obs", *tasks.PRIOR_DESIGNS}
    assert set(df["D"]) == {4}
    assert {"q", "t"} <= set(df_t.columns)
    assert df["kl_oracle_model"].notna().all()
    # the uniform prior covers all 543 DAGs on 4 nodes
    assert df["true_graph_in_prior"].all()
    assert np.allclose(df["log_prior_true_graph"], -np.log(543))


def test_rejects_large_d():
    with pytest.raises(ValueError):
        run(
            4, 1, [0, 1], normal_predictor, max_oracle_nodes=3,
            log_graph_prior=uniform_prior(4),
        )  # fmt: skip


def test_default_graph_prior_covers_true_graphs_d3():
    # log_graph_prior=None: run_kshot_prior's own default (20000 graphs)
    df, _ = run(3, 3, [0, 1], normal_predictor, log_graph_prior=None)
    assert {"log_prior_true_graph", "true_graph_in_prior"} <= set(df.columns)
    assert df["true_graph_in_prior"].all()
    assert np.isfinite(df["log_prior_true_graph"]).all()
    per_task = df.groupby("task")["log_prior_true_graph"].nunique()
    assert (per_task == 1).all()


# --------------------------------------------------------------------------- #
# oracle parameter cache
# --------------------------------------------------------------------------- #
def fitted_oracle(D: int = 3) -> LinGaussOracle:
    """Oracle fitted on a fixed context: 30 obs rows + 4 rows of do(X0) from
    prior task 1. Uses the uniform graph prior: these tests compare the
    oracle with itself, so any graph prior will do."""
    scm, _, do_dist, _ = tasks.prior_scm(CFG, 1, SEED, D)
    rng = np.random.default_rng(0)
    x, m = scm.sample(30, None, None, rng)
    px, pm, _ = tasks.node_pools(scm, 4, do_dist, rng)[0]
    x, m = np.concatenate([x, px]), np.concatenate([m, pm])
    return LinGaussOracle(
        D,
        log_graph_prior=uniform_prior(D),
        prior=LinearGaussianPrior.from_config(CFG),
    ).fit(x, m)


def test_oracle_param_cache_matches_uncached():
    D = 3
    orc = fitted_oracle(D)
    cache: dict = {}  # shared by both budgets: their keys must not clash
    for _ in range(2):  # second pass: every draw served from the cache
        for budget in ("per_dag", "total"):
            kw = dict(n_mc=40, min_graph_prob=1e-4, seed=3, budget=budget)
            for T in range(D):
                for Y in range(D):
                    if T == Y:
                        continue
                    plain = orc.predictive_components(T, Y, **kw)
                    cached = orc.predictive_components(
                        T, Y, param_cache=cache, **kw
                    )
                    for a, b in zip(plain, cached):
                        assert np.array_equal(a, b)
    assert (3, 40, 1e-4, "total") in cache
    assert any(isinstance(k[3], int) for k in cache)


# --------------------------------------------------------------------------- #
# oracle MC budget "total"
# --------------------------------------------------------------------------- #
def test_total_budget_components():
    orc = fitted_oracle()
    n_mc, min_p = 500, 0.02
    log_w, slope, var = orc.predictive_components(
        0, 2, n_mc=n_mc, min_graph_prob=min_p, seed=1, budget="total"
    )
    assert len(log_w) == len(slope) == len(var) == n_mc
    assert np.exp(log_w).sum() == pytest.approx(1.0)
    draws = orc._draw_total(n_mc, min_p, 1)
    assert sum(len(B) for _, B, _ in draws) == n_mc
    assert all(len(B) > 0 for _, B, _ in draws)
    post = orc.graph_post
    assert all(post[g] >= min_p for g, _, _ in draws)
    assert (post < min_p).any()  # the threshold drops some DAGs here


def test_total_and_per_dag_budgets_agree():
    orc = fitted_oracle()
    t = np.array([-1.5, 0.0, 1.5])
    out = {}
    for budget, n_mc in (("per_dag", 4000), ("total", 20000)):
        log_w, slope, var = orc.predictive_components(
            0, 2, n_mc=n_mc, min_graph_prob=1e-4, seed=2, budget=budget
        )
        _, w, var = oracle_weights(log_w, var)
        out[budget] = oracle_moments(w, slope, var, t)
    s_pd, _, mean_pd, var_pd = out["per_dag"]
    s_tot, _, mean_tot, var_tot = out["total"]
    assert s_tot == pytest.approx(s_pd, rel=0.05, abs=0.05)
    np.testing.assert_allclose(mean_tot, mean_pd, rtol=0.05, atol=0.08)
    np.testing.assert_allclose(var_tot, var_pd, rtol=0.1)


@pytest.mark.parametrize("chunk", [1, 128, 300, 1000, 5000])
def test_mix_logpdf_chunked_matches_unchunked(chunk):
    g = torch.Generator().manual_seed(0)
    n_t, n_y, K = 5, 7, 1000

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    mu, sd = randn(n_t, K), randn(n_t, K).exp()
    logw = torch.log_softmax(randn(n_t, K), dim=-1)
    y = 2 * randn(n_t, n_y)
    torch.testing.assert_close(
        mix_logpdf(y, mu, sd, logw, chunk=chunk), mix_logpdf(y, mu, sd, logw)
    )


# --------------------------------------------------------------------------- #
# model evaluation without the oracle
# --------------------------------------------------------------------------- #
def forbid_the_oracle(monkeypatch):
    """Make any use of the oracle or of its graph prior fail."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the oracle must not be used")

    monkeypatch.setattr(experiments, "LinGaussOracle", forbidden)
    monkeypatch.setattr(experiments, "empirical_graph_prior", forbidden)


def test_model_scores_do_not_depend_on_the_oracle(monkeypatch):
    D = 3
    predict = untrained_model_predictor(D)
    kw = dict(flag_ablation=True)
    # log_graph_prior=None: the oracle run estimates its own graph prior,
    # which reseeds the global generators; the tasks must not depend on it
    with_oracle = run(D, 3, [0, 2], predict, log_graph_prior=None, **kw)
    forbid_the_oracle(monkeypatch)
    model_only = run(
        D, 3, [0, 2], predict, log_graph_prior=None, with_oracle=False, **kw
    )
    for both, alone, oracle_columns in zip(
        with_oracle,
        model_only,
        (
            [*ORACLE_COLUMNS, "kl_oracle_model_noflag"],
            ORACLE_COLUMNS_BY_DO_VALUE,
        ),
    ):
        # same layout: identical columns in identical order
        assert list(alone.columns) == list(both.columns)
        assert both[oracle_columns].notna().all().all()
        assert alone[oracle_columns].isna().all().all()
        # every other column, the model scores among them, is exactly equal
        pd.testing.assert_frame_equal(
            alone.drop(columns=oracle_columns),
            both.drop(columns=oracle_columns),
            check_exact=True,
        )
    df = model_only[0]
    assert np.isfinite(df[["kl_truth_model", "slope_model"]]).all().all()
    assert df["kl_truth_model"].nunique() > 1


def flag_sensitive_predictor(x, m, Ts, Ys, grid, zero_flags=False):
    """N(0, 1) with the intervention flags, N(0.5, 4) with them zeroed."""
    mu, sd, logw = normal_predictor(x, m, Ts, Ys, grid)
    return (mu + 0.5, 2.0 * sd, logw) if zero_flags else (mu, sd, logw)


def flags_always_zeroed_predictor(x, m, Ts, Ys, grid, zero_flags=False):
    """What `flag_sensitive_predictor` predicts with the flags zeroed."""
    return flag_sensitive_predictor(x, m, Ts, Ys, grid, zero_flags=True)


@pytest.mark.parametrize("with_oracle", [True, False])
def test_flag_ablation_against_truth(with_oracle):
    kw = dict(log_graph_prior=uniform_prior(3), with_oracle=with_oracle)
    df, _ = run(3, 2, [0, 2], flag_sensitive_predictor, flag_ablation=True, **kw)
    # the flag-ablated model, scored as the model on the same truth samples
    zeroed, _ = run(3, 2, [0, 2], flags_always_zeroed_predictor, **kw)
    assert np.array_equal(df["kl_truth_model_noflag"], zeroed["kl_truth_model"])
    assert (df["kl_truth_model_noflag"] != df["kl_truth_model"]).all()
    assert df["kl_oracle_model_noflag"].notna().all() == with_oracle
    # neither flag-ablation column without flag_ablation
    assert not [c for c in zeroed.columns if c.endswith("noflag")]


def test_model_only_five_variables(monkeypatch):
    D = 5
    forbid_the_oracle(monkeypatch)
    df, df_t = run_kshot_prior(
        untrained_model_predictor(D),
        CFG_FIVE_VARIABLES,
        D=D,
        device="cpu",
        k_list=[0, 2],
        n_tasks=2,
        base_seed=SEED,
        n_y=16,
        n_t=5,
        with_oracle=False,
    )
    # obs + 4 designs at k = 2, for each of the D (D - 1) ordered pairs
    assert len(df) == 2 * D * (D - 1) * 5
    assert len(df_t) == len(df) * 5
    assert set(df["design"]) == {"obs", *tasks.PRIOR_DESIGNS}
    assert (df[df["design"] == "do(other)"]["n_other"] == D - 2).all()
    model_columns = ["kl_truth_model", "slope_truth", "slope_model"]
    assert np.isfinite(df[model_columns]).all().all()
    assert np.isfinite(
        df_t[["kl_truth_model", "mean_truth", "mean_model", "sd_model"]]
    ).all().all()  # fmt: skip
    assert df[ORACLE_COLUMNS].isna().all().all()
    assert df_t[ORACLE_COLUMNS_BY_DO_VALUE].isna().all().all()


def test_large_d_error_points_to_the_model_only_mode():
    with pytest.raises(ValueError, match="with_oracle=False"):
        run_kshot_prior(
            normal_predictor,
            CFG_FIVE_VARIABLES,
            D=5,
            device="cpu",
            k_list=[0, 1],
            n_tasks=1,
        )
