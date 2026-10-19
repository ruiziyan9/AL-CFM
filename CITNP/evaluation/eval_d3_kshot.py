"""
K-shot evaluation of the D=3 mixed-context model against the TRUTH and the
EXACT BAYESIAN ORACLE on a confounder SCM, for all six marginal interventional
queries X_Y | do(X_T), T != Y.

Ground truth: the standard confounder SCM, standardised to unit marginal
variances (variance ordering carries no information about direction):

    treatment  X0      confounder  X1      outcome  X2
    X1 -> X0 (0.9),  X1 -> X2 (0.8),  X0 -> X2 (-0.3)     raw weights
    noise variances (X0, X1, X2) = (0.2, 0.3, 0.25)        raw scale

The skeleton is complete, so observational data orients no edge: every
interventional query is non-identifiable from observational data alone.
X2 | do(X0) is confounded with a sign flip (true effect about -0.34,
naive observational slope about +0.21). With --random_weights each seed draws
fresh weights for the same graph instead.

Experiments (k = number of interventional rows added to n_obs observational
rows; contexts are nested in k, so curves over k are paired within a seed):
    random          each row intervenes on a uniformly random node
    do(treatment)   every row is do(X0)
    do(confounder)  every row is do(X1)
    do(outcome)     every row is do(X2)
k = 0 (observational only) is shared by all four.

For every context, query (T, Y) and do-value t on a fixed grid, three
predictive distributions are compared: truth p* (closed form), oracle p_B
(LinGaussOracle posterior predictive, enumerating all 25 DAGs) and model p_m:
    kl_truth_model    KL(p*  || p_m)   how wrong the model is about the world
    kl_truth_oracle   KL(p*  || p_B)   the best achievable from this context
    kl_oracle_model   KL(p_B || p_m)   distance from the Bayes answer
plus implied effects (slope of the predictive mean in t) and variance ratios.

query_type (large-k status under the nodes intervened on so far):
    seen / unseen_identified / unidentified

Outputs:
    <out>                one row per (seed, design, k, query), averaged over t
    <out>_by_t.csv       one row per (seed, design, k, query, t)
    <out>_main.png       KL vs number of experiments, 4 curves, oracle dashed
    <out>_by_query.png   the same per query
    <out>_effect.png     implied-effect error vs number of experiments

    python -m CITNP.evaluation.eval_d3_kshot \
        --ckpt <run_dir>/best_model.pt \
        --model_config CITNP/experiments/configs/oracle_d3_model_s.yaml \
        --data_config  CITNP/experiments/configs/oracle_d3.yaml \
        --out kshot.csv
    python -m CITNP.evaluation.eval_d3_kshot --plot_only kshot.csv   # re-plot

The oracle is Bayes-optimal only for the prior the model was trained on
(`linearnonidentifiable`, normalise: false).
"""

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

TREAT, CONF, OUTC = 0, 1, 2
ROLE = {TREAT: "treatment", CONF: "confounder", OUTC: "outcome"}
D = 3
QUERIES = [(T, Y) for T in range(D) for Y in range(D) if T != Y]  # (treatment, outcome)
MAIN_QUERY = f"{ROLE[OUTC]}|do({ROLE[TREAT]})"
DESIGNS = ["random", "do(treatment)", "do(confounder)", "do(outcome)"]
FIXED_NODE = {"do(treatment)": TREAT, "do(confounder)": CONF, "do(outcome)": OUTC}
COLORS = {"random": "C0", "do(treatment)": "C1", "do(confounder)": "C2", "do(outcome)": "C3"}


def qname(T, Y):
    return f"{ROLE[Y]}|do({ROLE[T]})"


# --------------------------------------------------------------------------- #
# Identifiability labels (confounder graph, true order X1, X0, X2)
# --------------------------------------------------------------------------- #
_ALL = frozenset(QUERIES)


def identified_queries(nodes: frozenset) -> frozenset:
    """Queries whose effect is identified, as k grows, by interventions on
    `nodes` (plus observational data), for the confounder graph."""
    if TREAT in nodes or {CONF, OUTC} <= nodes:
        return _ALL  # the causal order X1, X0, X2 is pinned down
    if nodes == {CONF}:
        # X1 first; X0 - X2 orientation open
        return frozenset({(CONF, TREAT), (CONF, OUTC), (TREAT, CONF), (OUTC, CONF)})
    if nodes == {OUTC}:
        # X2 last; X0 - X1 orientation open
        return frozenset({(OUTC, TREAT), (OUTC, CONF)})
    return frozenset()


def query_type(nodes: frozenset, T: int, Y: int) -> str:
    if T in nodes:
        return "seen"
    if (T, Y) in identified_queries(nodes):
        return "unseen_identified"
    return "unidentified"


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def standardise(scm):
    """Rescale every variable to unit marginal (observational) variance.
    x'_j = x_j / sd_j  =>  B'[i, j] = B[i, j] sd_i / sd_j,  s2'_j = s2_j / sd_j^2."""
    from CITNP.evaluation.oracle import LinearSCM

    A = np.linalg.inv(np.eye(D) - scm.B.T)
    sd = np.sqrt(np.diag(A @ np.diag(scm.s2) @ A.T))
    return LinearSCM(scm.B * sd[:, None] / sd[None, :], scm.s2 / sd**2)


def confounder_scm(b_ct=0.9, b_cy=0.8, b_ty=-0.3, s2=(0.2, 0.3, 0.25)):
    from CITNP.evaluation.oracle import LinearSCM

    B = np.zeros((D, D))
    B[CONF, TREAT], B[CONF, OUTC], B[TREAT, OUTC] = b_ct, b_cy, b_ty
    return standardise(LinearSCM(B, np.asarray(s2, dtype=float)))


def random_confounder_scm(rng):
    sign = lambda: rng.choice([-1.0, 1.0])
    return confounder_scm(
        b_ct=sign() * rng.uniform(0.5, 1.5),
        b_cy=sign() * rng.uniform(0.5, 1.5),
        b_ty=sign() * rng.uniform(0.2, 1.0),
        s2=rng.uniform(0.1, 1.0, D),
    )


def obs_cov(scm):
    A = np.linalg.inv(np.eye(D) - scm.B.T)
    return A @ np.diag(scm.s2) @ A.T


# --------------------------------------------------------------------------- #
# Densities
# --------------------------------------------------------------------------- #
def mix_logpdf(y, mu, sd, logw):
    """log density of y [n_t, n_y] under per-t Gaussian mixtures
    mu, sd, logw [n_t, K]  ->  [n_t, n_y]."""
    lp = torch.distributions.Normal(mu[:, None, :], sd[:, None, :]).log_prob(y[:, :, None])
    return torch.logsumexp(lp + logw[:, None, :], dim=-1)


@torch.no_grad()
def model_mixtures(model, ctx_x, ctx_int, grid, device, zero_flags=False):
    """Model predictive for all QUERIES at every do-value in grid, raw scale.
    The six queries share the context and run as one batch of six elements.
    Returns mu, sd, logw, each [Q, n_t, K] float64 on device."""
    from CITNP.trainer.mixed_context_trainer import _normalisation_stats

    Q = len(QUERIES)
    cx = torch.tensor(ctx_x, dtype=torch.float32)[None]
    ci = torch.tensor(ctx_int, dtype=torch.bool)[None]
    mean, std = _normalisation_stats(cx, ci)
    mean, std = mean[0].float(), std[0].float()
    cxn = ((cx - mean) / std)[..., None].repeat(Q, 1, 1, 1)
    flags = (torch.zeros_like(ci) if zero_flags else ci).repeat(Q, 1, 1)
    Ts = torch.tensor([T for T, _ in QUERIES])
    Ys = torch.tensor([Y for _, Y in QUERIES])
    t = torch.tensor(grid, dtype=torch.float32)
    tgt = torch.zeros(Q, len(grid), D, 1)
    for qi, (T, _) in enumerate(QUERIES):
        tgt[qi, :, T, 0] = (t - mean[T]) / std[T]
    out = model(
        context_data=cxn.to(device),
        target_data=tgt.to(device),
        intervention_index=Ts.to(device),
        outcome_index=Ys.to(device),
        context_int_mask=flags.to(device),
    )
    mu = out.pred_mean[:, :, 0].double()  # [Q, n_t, K], standardised scale
    sd = out.pred_std[:, :, 0].double()
    w = out.weights[:, :, 0].double()
    m_y = mean[Ys].double().to(device)[:, None, None]
    s_y = std[Ys].double().to(device)[:, None, None]
    logw = torch.log(w + 1e-12)
    logw = logw - torch.logsumexp(logw, dim=-1, keepdim=True)
    return mu * s_y + m_y, sd * s_y + 1e-12, logw


# --------------------------------------------------------------------------- #
# Experiment pools (nested in k)
# --------------------------------------------------------------------------- #
def make_pools(scm, k_max, do_scale, rng):
    """For each design: (x [k_max, D], int_mask [k_max, D], target node per row)."""
    pools = {}
    for design, node in FIXED_NODE.items():
        dv = rng.normal(0, do_scale, k_max)
        x, m = scm.sample(k_max, node, dv, rng)
        pools[design] = (x, m, np.full(k_max, node))
    seq = rng.integers(0, D, k_max)
    dv = rng.normal(0, do_scale, k_max)
    rx, rm = [], []
    for i in range(k_max):
        xi, mi = scm.sample(1, int(seq[i]), dv[i : i + 1], rng)
        rx.append(xi)
        rm.append(mi)
    pools["random"] = (np.concatenate(rx), np.concatenate(rm), seq)
    return pools


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _curve(df, design, cols, query=None):
    """Mean and s.e. over seeds of cols vs k, with k = 0 prepended; averages
    over queries within a seed first when query is None."""
    d = df if query is None else df[df["query"] == query]
    h = pd.concat([d[d.k == 0], d[d.design == design]])
    per_seed = h.groupby(["seed", "k"])[cols].mean()
    g = per_seed.groupby("k")
    return g.mean(), g.sem()


def plot_results(df, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xlab = "number of experiments (interventional samples)"

    # 1. main figure: 4 curves, model solid, oracle dashed
    panels = [
        ("kl_truth_model", "kl_truth_oracle", MAIN_QUERY, f"KL(truth || ·): {MAIN_QUERY}"),
        ("kl_truth_model", "kl_truth_oracle", None, "KL(truth || ·): mean of 6 queries"),
        ("kl_oracle_model", None, None, "KL(Bayes oracle || model): mean of 6 queries"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (mcol, ocol, q, title) in zip(axes, panels):
        for d in DESIGNS:
            cols = [mcol] + ([ocol] if ocol else [])
            mu, se = _curve(df, d, cols, q)
            ax.plot(mu.index, mu[mcol], "-o", ms=3, color=COLORS[d], label=f"model, {d}")
            ax.fill_between(mu.index, mu[mcol] - se[mcol], mu[mcol] + se[mcol],
                            color=COLORS[d], alpha=0.15)
            if ocol:
                ax.plot(mu.index, mu[ocol], "--", color=COLORS[d], label=f"oracle, {d}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(sorted(df.k.unique()))
        ax.set_xlabel(xlab)
        ax.set_title(title)
    axes[0].set_ylabel("KL, nats per query point")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(prefix + "_main.png", dpi=150)
    plt.close(fig)

    # 2. per query: KL to truth, model solid, oracle dashed
    queries = [qname(T, Y) for T, Y in QUERIES]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for ax, q in zip(axes.flat, queries):
        for d in DESIGNS:
            mu, _ = _curve(df, d, ["kl_truth_model", "kl_truth_oracle"], q)
            ax.plot(mu.index, mu.kl_truth_model, "-o", ms=3, color=COLORS[d], label=f"model, {d}")
            ax.plot(mu.index, mu.kl_truth_oracle, "--", color=COLORS[d], label=f"oracle, {d}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"KL(truth || ·): {q}")
    for ax in axes[1]:
        ax.set_xlabel(xlab)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(prefix + "_by_query.png", dpi=150)
    plt.close(fig)

    # 3. implied causal effect error (model solid, oracle dashed, naive dotted)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for ax, q in zip(axes.flat, queries):
        for d in DESIGNS:
            mu, _ = _curve(df, d, ["slope_err_model_truth", "slope_err_oracle_truth",
                                   "slope_err_naive_truth"], q)
            ax.plot(mu.index, mu.slope_err_model_truth, "-o", ms=3, color=COLORS[d],
                    label=f"model, {d}")
            ax.plot(mu.index, mu.slope_err_oracle_truth, "--", color=COLORS[d],
                    label=f"oracle, {d}")
        ax.axhline(mu.slope_err_naive_truth.iloc[0], color="r", ls=":", lw=1,
                   label="naive obs regression")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"effect error (estimate - truth): {q}")
    for ax in axes[1]:
        ax.set_xlabel(xlab)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(prefix + "_effect.png", dpi=150)
    plt.close(fig)
    print(f"wrote {prefix}_main.png, {prefix}_by_query.png, {prefix}_effect.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--model_config")
    ap.add_argument("--data_config")
    ap.add_argument("--plot_only", default=None, help="CSV from a previous run: only plot")
    ap.add_argument("--n_seeds", type=int, default=100)
    ap.add_argument("--n_obs", type=int, default=128)
    ap.add_argument("--k_max", type=int, default=10)
    ap.add_argument("--k_list", default=None,
                    help="comma-separated numbers of experiments, e.g. 0,1,3,5,10; "
                         "overrides --k_max (default: 0..k_max)")
    ap.add_argument("--random_weights", action="store_true",
                    help="fresh confounder-SCM weights per seed (same graph)")
    ap.add_argument("--do_scale", type=float, default=1.5,
                    help="sd of do-values in the context's interventional rows")
    ap.add_argument("--t_max", type=float, default=3.0, help="query grid is [-t_max, t_max]")
    ap.add_argument("--n_t", type=int, default=13, help="query grid points")
    ap.add_argument("--n_y", type=int, default=64, help="MC samples per grid point for KLs")
    ap.add_argument("--n_mc", type=int, default=200, help="oracle parameter draws per DAG")
    ap.add_argument("--min_graph_prob", type=float, default=1e-4)
    ap.add_argument("--out", default="kshot.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    k_list = (sorted({int(k) for k in a.k_list.split(",")}) if a.k_list
              else list(range(a.k_max + 1)))
    a.k_max = max(k_list)

    if a.plot_only:
        plot_results(pd.read_csv(a.plot_only), a.plot_only.removesuffix(".csv"))
        return
    if not (a.ckpt and a.model_config and a.data_config):
        ap.error("--ckpt, --model_config and --data_config are required")

    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.evaluation.eval_d3_oracle import empirical_graph_prior, load_model
    from CITNP.evaluation.oracle import LinGaussOracle

    dev = a.device
    data_cfg = load_config(a.data_config)
    lo, hi = data_cfg.n_ctx_range
    if not (lo <= a.n_obs and a.n_obs + a.k_max <= hi):
        print(f"WARNING: context sizes {a.n_obs}..{a.n_obs + a.k_max} "
              f"outside training n_ctx_range {data_cfg.n_ctx_range}")

    model = load_model(a.ckpt, a.model_config, dev)
    prior = empirical_graph_prior(data_cfg)

    fixed = confounder_scm()
    if not a.random_weights:
        S = obs_cov(fixed)
        print(f"true SCM (standardised) B =\n{np.round(fixed.B, 3)}\ns2 = {np.round(fixed.s2, 3)}")
        for T, Y in QUERIES:
            b, v = fixed.do_dist(T, Y)
            print(f"  {qname(T, Y):26s} N({b:+.3f} t, {v:.3f})   "
                  f"naive obs slope {S[T, Y] / S[T, T]:+.3f}")

    grid = np.linspace(-a.t_max, a.t_max, a.n_t)
    g = torch.tensor(grid, dtype=torch.float64, device=dev)
    contexts = [("obs", 0)] + [(d, k) for d in DESIGNS for k in k_list if k > 0]

    rows, rows_t = [], []
    for s in range(a.n_seeds):
        rng = np.random.default_rng([s, 7])
        scm = random_confounder_scm(rng) if a.random_weights else fixed
        S = obs_cov(scm)
        truth = {q: scm.do_dist(*q) for q in QUERIES}
        naive = {(T, Y): float(S[T, Y] / S[T, T]) for T, Y in QUERIES}

        obs_x, obs_m = scm.sample(a.n_obs, None, None, rng)
        pools = make_pools(scm, a.k_max, a.do_scale, rng)

        for c_i, (design, k) in enumerate(contexts):
            if k == 0:
                x, m, nodes = obs_x, obs_m, frozenset()
            else:
                px, pm, seq = pools[design]
                x = np.concatenate([obs_x, px[:k]])
                m = np.concatenate([obs_m, pm[:k]])
                nodes = frozenset(int(j) for j in seq[:k])
            perm = np.random.default_rng([s, c_i, k]).permutation(len(x))
            x, m = x[perm], m[perm]

            orc = LinGaussOracle(D, prior).fit(x, m)
            mod = model_mixtures(model, x, m, grid, dev)
            mod_nf = model_mixtures(model, x, m, grid, dev, zero_flags=True)
            qrng = np.random.default_rng([s, c_i, k, 99])

            for qi, (T, Y) in enumerate(QUERIES):
                # truth p*
                b, v = truth[(T, Y)]
                y_true = b * grid[:, None] + np.sqrt(v) * qrng.standard_normal((a.n_t, a.n_y))
                y_true = torch.tensor(y_true, device=dev)
                lp_true = torch.distributions.Normal(b * g[:, None], np.sqrt(v)).log_prob(y_true)

                # exact Bayes p_B
                log_w, slope, var = orc.predictive_components(
                    T, Y, n_mc=a.n_mc, min_graph_prob=a.min_graph_prob, seed=s
                )
                log_w = log_w - logsumexp(log_w)
                w = np.exp(log_w)
                var = np.maximum(var, 1e-12)
                idx = qrng.choice(len(w), size=(a.n_t, a.n_y), p=w)
                y_or = slope[idx] * grid[:, None] + np.sqrt(var[idx]) * qrng.standard_normal(
                    (a.n_t, a.n_y)
                )
                y_or = torch.tensor(y_or, device=dev)
                o_mu = torch.tensor(slope, device=dev)[None] * g[:, None]
                o_sd = torch.tensor(np.sqrt(var), device=dev)[None].expand_as(o_mu)
                o_lw = torch.tensor(log_w, device=dev)[None].expand_as(o_mu)

                m_par = tuple(p[qi] for p in mod)
                nf_par = tuple(p[qi] for p in mod_nf)

                lp_o_on_o = mix_logpdf(y_or, o_mu, o_sd, o_lw)
                kl_om = (lp_o_on_o - mix_logpdf(y_or, *m_par)).mean(1)
                kl_om_nf = (lp_o_on_o - mix_logpdf(y_or, *nf_par)).mean(1)
                kl_tm = (lp_true - mix_logpdf(y_true, *m_par)).mean(1)
                kl_to = (lp_true - mix_logpdf(y_true, o_mu, o_sd, o_lw)).mean(1)
                kl_om, kl_om_nf, kl_tm, kl_to = (
                    z.cpu().numpy() for z in (kl_om, kl_om_nf, kl_tm, kl_to)
                )

                # moments
                s_bar = float(w @ slope)
                s_sd = float(np.sqrt(max(w @ slope**2 - s_bar**2, 0.0)))
                mean_o = s_bar * grid
                var_o = float(w @ var) + grid**2 * s_sd**2
                mu_m, sd_m, lw_m = (p.cpu().numpy() for p in m_par)
                w_m = np.exp(lw_m)
                mean_m = (w_m * mu_m).sum(-1)
                var_m = (w_m * (sd_m**2 + mu_m**2)).sum(-1) - mean_m**2
                slope_m, icpt_m = np.polyfit(grid, mean_m, 1)

                q = qname(T, Y)
                qtype = query_type(nodes, T, Y)
                rows.append({
                    "seed": s, "design": design, "k": k, "T": T, "Y": Y,
                    "query": q, "query_type": qtype,
                    "intervened_nodes": "".join(str(j) for j in sorted(nodes)),
                    "effect": "zero" if abs(b) < 1e-9 else "nonzero",
                    "kl_truth_model": float(kl_tm.mean()),
                    "kl_truth_oracle": float(kl_to.mean()),
                    "kl_oracle_model": float(kl_om.mean()),
                    "kl_oracle_model_noflag": float(kl_om_nf.mean()),
                    "excess_kl_truth": float((kl_tm - kl_to).mean()),
                    "slope_truth": float(b), "slope_naive_obs": naive[(T, Y)],
                    "slope_oracle": s_bar, "slope_sd_oracle": s_sd,
                    "slope_model": float(slope_m), "intercept_model": float(icpt_m),
                    "slope_err_model_truth": float(slope_m - b),
                    "slope_err_oracle_truth": float(s_bar - b),
                    "slope_err_naive_truth": float(naive[(T, Y)] - b),
                    "var_ratio_model_oracle": float(np.mean(var_m / var_o)),
                })
                for ti, tv in enumerate(grid):
                    rows_t.append({
                        "seed": s, "design": design, "k": k, "query": q,
                        "query_type": qtype, "t": float(tv),
                        "kl_truth_model": float(kl_tm[ti]),
                        "kl_truth_oracle": float(kl_to[ti]),
                        "kl_oracle_model": float(kl_om[ti]),
                        "mean_truth": float(b * tv), "sd_truth": float(np.sqrt(v)),
                        "mean_oracle": float(mean_o[ti]), "sd_oracle": float(np.sqrt(var_o[ti])),
                        "mean_model": float(mean_m[ti]),
                        "sd_model": float(np.sqrt(max(var_m[ti], 0.0))),
                    })
        if (s + 1) % 10 == 0:
            print(f"{s + 1}/{a.n_seeds} seeds done")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    prefix = a.out.removesuffix(".csv")
    pd.DataFrame(rows_t).to_csv(prefix + "_by_t.csv", index=False)
    print(f"wrote {a.out} and {prefix}_by_t.csv")

    cols = ["kl_truth_model", "kl_truth_oracle", "kl_oracle_model", "kl_oracle_model_noflag"]
    sel = df[df.k.isin(k_list)]
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(f"\n=== {MAIN_QUERY}, by experiment design and k")
        print(sel[sel["query"] == MAIN_QUERY].groupby(["design", "k"])[cols].mean().round(3))
        print("\n=== all 6 queries, by experiment design and k")
        print(sel.groupby(["design", "k"])[cols].mean().round(3))
        print("\n=== by query type and k")
        print(sel.groupby(["query_type", "k"])[cols].mean().round(3))

    plot_results(df, prefix)


if __name__ == "__main__":
    main()
