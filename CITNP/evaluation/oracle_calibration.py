"""Simulation-based calibration (SBC) of the exact oracle.

    python -m CITNP.evaluation calibrate --data_config CFG --n_tasks 2000 \
        --out calib.csv [--n_mc 1000] [--compare_mc]

Tasks are drawn from the prior the oracle assumes. If oracle and generator
agree, then for every query row
  * the PIT (probability integral transform: the oracle's predictive CDF at
    the true query value) is Uniform(0, 1), in every nT bucket;
  * the posterior probabilities the oracle gives to Markov equivalence classes
    (MECs) are calibrated: classes given probability p are the true one with
    frequency p.
`p_true_mec` is the posterior probability of the true graph's MEC (large on
average when the context identifies the structure, so it is not itself a
calibration measure; the reliability plot uses all MECs of every task).

The predictive is the oracle's Monte Carlo mixture, so PIT / NLL carry MC
noise; `compare_mc` also evaluates with 2x the draws and reports the mean
|delta NLL| per (task, T, Y).

Writes <out> (one row per query row), <stem>_mec.csv (one row per task and
MEC), <stem>_pit.png and <stem>_reliability.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from CITNP.datasets.linear_gaussian_prior import LinearGaussianPrior
from CITNP.evaluation.oracle import LinGaussOracle, all_dags, mec_key
from CITNP.evaluation.tasks import BUCKETS, empirical_graph_prior, n_int_bucket

__all__ = ["calibrate", "plot_pit", "plot_reliability", "summarise"]

COLUMNS = [
    "task",
    "T",
    "Y",
    "nT",
    "nT_bucket",
    "question_class",
    "pit",
    "p_true_mec",
    "nll",
]


def _pit_and_logpdf(orc, T, Y, t, y, n_mc, seed):
    """PIT and log density of y under the oracle's mixture p(Y | do(T = t))."""
    log_w, slope, var = orc.predictive_components(T, Y, n_mc=n_mc, seed=seed)
    log_w = log_w - logsumexp(log_w)  # graphs below min_graph_prob are dropped
    var = np.maximum(var, 1e-12)
    mu = slope[None] * t[:, None]
    z = (y[:, None] - mu) / np.sqrt(var)[None]
    pit = (np.exp(log_w)[None] * norm.cdf(z)).sum(1)
    lp = logsumexp(-0.5 * (np.log(2 * np.pi * var)[None] + z**2) + log_w[None], 1)
    return pit, lp


def calibrate(
    cfg,
    n_tasks: int,
    n_mc: int = 1000,
    base_seed: int = 7001,
    compare_mc: bool = False,
    D: int = 3,
    log_graph_prior: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Oracle PIT and MEC posterior on `n_tasks` tasks drawn from the prior.

    Args:
    ----------
    cfg : MixedRegimeConfig
        Data config (linear-Gaussian family, normalise: false).

    n_tasks : int
        Number of tasks.

    n_mc : int
        Oracle posterior draws per DAG.

    base_seed : int
        Task keys are make_key(SOURCE_ADHOC, 0, base_seed, i); the default
        differs from the oracle bank's seed.

    compare_mc : bool
        Also evaluate with 2 * n_mc draws (other seed); reports mean |delta NLL|.

    log_graph_prior : dict | None
        dag_key -> log prior of the oracle (default: `empirical_graph_prior`).

    Returns:
    ----------
    (rows, mec, info): `rows` has one row per query row (COLUMNS), `mec` one
    row per (task, MEC) with columns task, mec, p, is_true, and `info` holds
    `mc_abs_dnll` (None unless compare_mc).
    """
    from CITNP.datasets.mixed_regime_generator import (
        SOURCE_ADHOC,
        generate_task,
        make_key,
    )
    from CITNP.prior_diagnostics.stats import question_class

    prior = LinearGaussianPrior.from_config(cfg)
    if log_graph_prior is None:
        log_graph_prior = empirical_graph_prior(cfg, num_nodes=D)
    dags = all_dags(D)
    mec_of = [mec_key(g) for g in dags]
    mec_ids = {k: i for i, k in enumerate(dict.fromkeys(mec_of))}
    dag_mec = np.array([mec_ids[k] for k in mec_of])

    rows, mec_rows, dnll = [], [], []
    for i in range(n_tasks):
        task = generate_task(
            cfg, make_key(SOURCE_ADHOC, 0, base_seed, i), num_variables=D
        )
        ctx_x = task.ctx_x.astype(np.float64)
        ctx_int = task.ctx_int.astype(bool)
        qx = task.qry_x.astype(np.float64)
        qi = task.qry_int.astype(bool)
        graph = task.graph.astype(int)
        orc = LinGaussOracle(D, log_graph_prior=log_graph_prior, prior=prior).fit(
            ctx_x, ctx_int
        )

        p_mec = np.bincount(
            dag_mec, weights=orc.graph_post, minlength=len(mec_ids)
        )
        true_mec = mec_ids[mec_key(graph)]
        for m, p in enumerate(p_mec):
            mec_rows.append(
                dict(task=i, mec=m, p=float(p), is_true=int(m == true_mec))
            )

        for T in sorted({int(np.argmax(r)) for r in qi}):
            sel = qi[:, T]
            t = qx[sel, T]
            n_T = int(ctx_int[:, T].sum())
            for Y in range(D):
                if Y == T:
                    continue
                y = qx[sel, Y]
                pit, lp = _pit_and_logpdf(orc, T, Y, t, y, n_mc, seed=0)
                if compare_mc:
                    _, lp2 = _pit_and_logpdf(orc, T, Y, t, y, 2 * n_mc, seed=1)
                    dnll.append(abs(float(lp.mean() - lp2.mean())))
                qc = question_class(graph, T, Y)
                for k in range(len(y)):
                    rows.append(
                        dict(
                            task=i,
                            T=T,
                            Y=Y,
                            nT=n_T,
                            nT_bucket=n_int_bucket(n_T),
                            question_class=qc,
                            pit=float(pit[k]),
                            p_true_mec=float(p_mec[true_mec]),
                            nll=float(-lp[k]),
                        )  # fmt: skip
                    )
    info = {"mc_abs_dnll": float(np.mean(dnll)) if dnll else None}
    return pd.DataFrame(rows, columns=COLUMNS), pd.DataFrame(mec_rows), info


def plot_pit(df: pd.DataFrame, out: str, bins: int = 20) -> None:
    """PIT histogram per nT bucket (flat at density 1 when calibrated)."""
    import matplotlib.pyplot as plt

    from CITNP.evaluation.plots import _save

    fig, axes = plt.subplots(
        1, len(BUCKETS), figsize=(4 * len(BUCKETS), 3), sharey=True
    )
    for ax, b in zip(axes, BUCKETS):
        v = df.loc[df.nT_bucket == b, "pit"].to_numpy()
        ax.hist(v, bins=bins, range=(0, 1), density=True, color="#3182BD")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_title(f"{b} (n={len(v)})")
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("density")
    _save(fig, out)


def plot_reliability(mec: pd.DataFrame, out: str, bins: int = 10) -> None:
    """Reliability of the oracle's MEC posterior: predicted p vs how often that
    MEC is the true one, over all (task, MEC) pairs."""
    import matplotlib.pyplot as plt

    from CITNP.evaluation.plots import _save

    fig, ax = plt.subplots(figsize=(4, 4))
    rel = reliability_table(mec, bins)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(rel.p_mean, rel.freq, "o-", color="#3182BD")
    ax.set_xlabel("posterior probability of MEC")
    ax.set_ylabel("fraction that is the true MEC")
    _save(fig, out)


def reliability_table(mec: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Per probability bin: mean predicted p, observed frequency, count."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(mec.p, edges[1:-1]), 0, bins - 1)
    g = mec.assign(bin=b).groupby("bin")
    return pd.DataFrame(
        {"p_mean": g.p.mean(), "freq": g.is_true.mean(), "n": g.size()}
    ).reset_index(drop=True)


def summarise(df: pd.DataFrame, mec: pd.DataFrame, info: dict) -> dict:
    """Headline calibration numbers: per-bucket max deviation of the 10-bin PIT
    histogram from 1, the largest reliability gap (bins with >= 100 pairs),
    mean p_true_mec and the MC-noise |delta NLL|."""
    out: dict = {}
    for b in BUCKETS:
        v = df.loc[df.nT_bucket == b, "pit"].to_numpy()
        if len(v):
            h, _ = np.histogram(v, bins=10, range=(0, 1), density=True)
            out[f"pit_max_dev[{b}]"] = float(np.abs(h - 1).max())
    rel = reliability_table(mec)
    rel = rel[rel.n >= 100]
    out["reliability_max_gap"] = float((rel.p_mean - rel.freq).abs().max())
    out["mean_p_true_mec"] = float(df.groupby("task").p_true_mec.first().mean())
    out["mc_abs_dnll"] = info["mc_abs_dnll"]
    return out


def run(a) -> None:
    """Body of `python -m CITNP.evaluation calibrate` (a: parsed arguments)."""
    import matplotlib

    matplotlib.use("Agg")
    from CITNP.datasets.create_mixed_regime_chunks import load_config

    cfg = load_config(a.data_config)
    df, mec, info = calibrate(
        cfg, a.n_tasks, n_mc=a.n_mc, base_seed=a.seed, compare_mc=a.compare_mc
    )
    out = Path(a.out)
    stem = out.with_suffix("")
    df.to_csv(out, index=False)
    mec.to_csv(f"{stem}_mec.csv", index=False)
    plot_pit(df, f"{stem}_pit.png")
    plot_reliability(mec, f"{stem}_reliability.png")
    print(f"wrote {out}, {stem}_mec.csv, {stem}_pit.png, {stem}_reliability.png")
    for k, v in summarise(df, mec, info).items():
        print(f"  {k:28s} {v}")
