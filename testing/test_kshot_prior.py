"""kshot-prior: tasks drawn from the training prior, nested k-shot contexts,
designs, and the identity E[excess] = E[KL(oracle||model)] with injected
predictors (CPU, no checkpoint)."""

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
from CITNP.evaluation import tasks
from CITNP.evaluation.experiments import _build_contexts, run_kshot_prior
from CITNP.evaluation.metrics import oracle_mixture_on_grid, oracle_weights
from CITNP.evaluation.oracle import LinGaussOracle, all_dags, dag_key
from CITNP.prior_diagnostics.linear_gaussian import scm_from_debug

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


def uniform_prior(D: int) -> dict:
    dags = all_dags(D)
    return {dag_key(g): -np.log(len(dags)) for g in dags}


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


def run(D, n_tasks, k_list, predict, n_mc=50, **kw):
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
        log_graph_prior=uniform_prior(D),
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
            make_key(SOURCE_ADHOC, 0, SEED, i),
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

    df, _ = run(D, 2, [0, 2, 4], spy)
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
    df, _ = run(
        D,
        40,
        [0, 3],
        make_oracle_predictor(D, uniform_prior(D), 200),
        n_mc=200,
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
    df, df_t = run(4, 2, [0, 2], normal_predictor)
    other = df[df["design"] == "do(other)"]
    assert len(other) > 0 and (other["n_other"] == 2).all()
    assert (df[df["design"] != "do(other)"]["n_other"] == 0).all()
    assert set(df["design"]) == {"obs", *tasks.PRIOR_DESIGNS}
    assert set(df["D"]) == {4}
    assert {"q", "t"} <= set(df_t.columns)
    assert df["kl_oracle_model"].notna().all()


def test_rejects_large_d():
    with pytest.raises(ValueError):
        run(4, 1, [0, 1], normal_predictor, max_oracle_nodes=3)


# --------------------------------------------------------------------------- #
# oracle parameter cache
# --------------------------------------------------------------------------- #
def test_oracle_param_cache_matches_uncached():
    D = 3
    scm, _, do_dist, _ = tasks.prior_scm(CFG, 1, SEED, D)
    rng = np.random.default_rng(0)
    x, m = scm.sample(30, None, None, rng)
    px, pm, _ = tasks.node_pools(scm, 4, do_dist, rng)[0]
    x, m = np.concatenate([x, px]), np.concatenate([m, pm])
    orc = LinGaussOracle(
        D,
        log_graph_prior=uniform_prior(D),
        prior=LinearGaussianPrior.from_config(CFG),
    ).fit(x, m)
    cache: dict = {}
    for _ in range(2):  # second pass: every DAG served from the cache
        for T in range(D):
            for Y in range(D):
                if T == Y:
                    continue
                kw = dict(n_mc=40, min_graph_prob=1e-4, seed=3)
                plain = orc.predictive_components(T, Y, **kw)
                cached = orc.predictive_components(
                    T, Y, param_cache=cache, **kw
                )
                for a, b in zip(plain, cached):
                    assert np.array_equal(a, b)
    assert len(cache) > 0
