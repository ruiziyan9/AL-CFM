"""Evaluation loops: tasks x model x oracle -> table of scores.

Each `run_*` takes a loaded model (see `model_api.load_model`) and returns
plain results (pandas DataFrames or a dict); writing files, printing summaries
and plotting are left to the caller.
"""

import numpy as np
import pandas as pd
import torch

from CITNP.evaluation.metrics import (
    implied_slope,
    kl_mc,
    mix_logpdf,
    mixture_moments,
    naive_slope,
    obs_cov,
    oracle_mixture_on_grid,
    oracle_moments,
    oracle_weights,
    sample_oracle,
    sample_truth,
    truth_logpdf,
    variance_ratio,
)
from CITNP.evaluation.model_api import (
    effective_components,
    mixture_cdf,
    mixture_logpdf,
    predict_mixture,
)
from CITNP.evaluation.oracle import LinGaussOracle
from CITNP.evaluation.tasks import (
    DESIGNS,
    QUERIES,
    confounder_scm,
    context_designs,
    empirical_graph_prior,
    make_context,
    make_pools,
    qname,
    query_type,
    random_confounder_scm,
    scm_suite,
)


# --------------------------------------------------------------------------- #
# K-shot evaluation on the confounder SCM
# --------------------------------------------------------------------------- #
def run_kshot(
    model: torch.nn.Module,
    data_cfg,
    *,
    device: str,
    k_list: list[int] | None = None,
    n_seeds: int = 100,
    n_obs: int = 128,
    random_weights: bool = False,
    do_scale: float = 1.5,
    t_max: float = 3.0,
    n_t: int = 13,
    n_y: int = 64,
    n_mc: int = 200,
    min_graph_prob: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-shot evaluation against the TRUTH and the EXACT BAYESIAN ORACLE on
    the confounder SCM, for all six queries X_Y | do(X_T), T != Y.

    Ground truth (`tasks.confounder_scm`), standardised to unit marginal
    variances: X1 -> X0 (0.9), X1 -> X2 (0.8), X0 -> X2 (-0.3), noise
    variances (0.2, 0.3, 0.25) on the raw scale; X0 = treatment, X1 =
    confounder, X2 = outcome. The skeleton is complete, so observational data
    orients no edge; X2 | do(X0) is confounded with a sign flip. With
    random_weights each seed draws fresh weights for the same graph.

    k = number of interventional rows added to n_obs observational rows,
    nested in k within a seed. Designs: random (each row intervenes on a
    uniformly random node), do(treatment), do(confounder), do(outcome); k = 0
    is shared by all four. For each context, query and do-value t on a grid,
    truth p*, oracle p_B (LinGaussOracle) and model p_m are compared:
        kl_truth_model    KL(p*  || p_m)   how wrong the model is about the world
        kl_truth_oracle   KL(p*  || p_B)   the best achievable from this context
        kl_oracle_model   KL(p_B || p_m)   distance from the Bayes answer
    plus implied effects (slope of the predictive mean in t) and variance
    ratios. query_type is seen / unseen_identified / unidentified. The oracle
    is Bayes-optimal only for the prior the model was trained on
    (`linearnonidentifiable`, normalise: false).

    k_list: values of k (default 0..10). Returns (one row per seed, design, k,
    query, averaged over the t grid; one row per seed, design, k, query, t).
    """
    k_list = list(range(11)) if k_list is None else sorted(set(k_list))
    k_max = max(k_list)
    dev = device
    lo, hi = data_cfg.n_ctx_range
    if not (lo <= n_obs and n_obs + k_max <= hi):
        print(f"WARNING: context sizes {n_obs}..{n_obs + k_max} "
              f"outside training n_ctx_range {data_cfg.n_ctx_range}")  # fmt: skip

    prior = empirical_graph_prior(data_cfg)

    fixed = confounder_scm()
    if not random_weights:
        S = obs_cov(fixed)
        print(f"true SCM (standardised) B =\n{np.round(fixed.B, 3)}\ns2 = {np.round(fixed.s2, 3)}")
        for T, Y in QUERIES:
            b, v = fixed.do_dist(T, Y)
            print(f"  {qname(T, Y):26s} N({b:+.3f} t, {v:.3f})   "
                  f"naive obs slope {naive_slope(S, T, Y):+.3f}")  # fmt: skip

    grid = np.linspace(-t_max, t_max, n_t)
    g = torch.tensor(grid, dtype=torch.float64, device=dev)
    Ts = np.array([T for T, _ in QUERIES])
    Ys = np.array([Y for _, Y in QUERIES])
    contexts = [("obs", 0)] + [(d, k) for d in DESIGNS for k in k_list if k > 0]

    rows, rows_t = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng([s, 7])
        scm = random_confounder_scm(rng) if random_weights else fixed
        S = obs_cov(scm)
        truth = {q: scm.do_dist(*q) for q in QUERIES}
        naive = {(T, Y): naive_slope(S, T, Y) for T, Y in QUERIES}

        obs_x, obs_m = scm.sample(n_obs, None, None, rng)
        pools = make_pools(scm, k_max, do_scale, rng)

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

            orc = LinGaussOracle(len(scm.s2), prior).fit(x, m)
            # all six queries share the context: one batch of six elements
            mod = predict_mixture(model, x, m, Ts, Ys, grid, device=dev)
            mod_nf = predict_mixture(model, x, m, Ts, Ys, grid, device=dev, zero_flags=True)
            qrng = np.random.default_rng([s, c_i, k, 99])

            for qi, (T, Y) in enumerate(QUERIES):
                # truth p*
                b, v = truth[(T, Y)]
                y_true = torch.tensor(sample_truth(b, v, grid, n_y, qrng), device=dev)
                lp_true = truth_logpdf(b, v, g, y_true)

                # exact Bayes p_B
                log_w, slope, var = orc.predictive_components(
                    T, Y, n_mc=n_mc, min_graph_prob=min_graph_prob, seed=s
                )
                log_w, w, var = oracle_weights(log_w, var)
                y_or = sample_oracle(w, slope, var, grid, n_y, qrng)
                y_or = torch.tensor(y_or, device=dev)
                o_par = oracle_mixture_on_grid(log_w, slope, var, g)

                m_par = tuple(p[qi] for p in mod)
                nf_par = tuple(p[qi] for p in mod_nf)

                lp_o_on_o = mix_logpdf(y_or, *o_par)
                kl_om = kl_mc(lp_o_on_o, mix_logpdf(y_or, *m_par))
                kl_om_nf = kl_mc(lp_o_on_o, mix_logpdf(y_or, *nf_par))
                kl_tm = kl_mc(lp_true, mix_logpdf(y_true, *m_par))
                kl_to = kl_mc(lp_true, mix_logpdf(y_true, *o_par))

                # moments
                s_bar, s_sd, mean_o, var_o = oracle_moments(w, slope, var, grid)
                mean_m, var_m = mixture_moments(*(p.cpu().numpy() for p in m_par))
                slope_m, icpt_m = implied_slope(grid, mean_m)

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
                    "var_ratio_model_oracle": variance_ratio(var_m, var_o),
                })  # fmt: skip
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
                    })  # fmt: skip
        if (s + 1) % 10 == 0:
            print(f"{s + 1}/{n_seeds} seeds done")

    return pd.DataFrame(rows), pd.DataFrame(rows_t)


# --------------------------------------------------------------------------- #
# Suite of fixed SCMs x context designs
# --------------------------------------------------------------------------- #
def run_scm_suite(
    model: torch.nn.Module,
    data_cfg,
    *,
    device: str,
    n_seeds: int = 30,
    n_query: int = 256,
    do_scale: float = 1.5,
    n_mc: int = 1000,
    num_nodes: int = 3,
) -> pd.DataFrame:
    """NLL per query point (raw scale) of truth, oracle and model, with and
    without intervention flags, for every SCM in `scm_suite()` and design in
    `context_designs()`; one row per (scm, design), mean and s.e. over seeds.

    NLLs are in nats per query row on the RAW data scale. The oracle is only
    Bayes-optimal for the prior the model was trained on
    (`linearnonidentifiable`, normalise: false); for other priors the truth
    columns are still exact and the oracle columns are a reference."""
    prior = empirical_graph_prior(data_cfg, num_nodes=num_nodes)

    rows = []
    for name, scm, (T, Y), why in scm_suite():
        if len(scm.s2) != num_nodes:
            raise ValueError(f"SCM {name} has {len(scm.s2)} nodes, expected {num_nodes}")
        slope, var = scm.do_dist(T, Y)
        print(f"\n=== {name}: Y=X{Y} | do(X{T}=t) ~ N({slope:.2f} t, {var:.3f})  [{why}]")
        for label, regimes in context_designs(num_nodes=num_nodes):
            r: dict[str, list] = {
                k: [] for k in ("truth", "oracle", "model", "model_noflag", "p_anc", "H")
            }
            for s in range(n_seeds):
                rng = np.random.default_rng(1000 * s + 7)
                x, m = make_context(scm, regimes, do_scale, rng)
                t = rng.normal(0, do_scale, n_query)
                xq, _ = scm.sample(n_query, T, t, rng)
                y = xq[:, Y]
                orc = LinGaussOracle(num_nodes, prior).fit(x, m)
                r["truth"].append(-scm.do_logpdf(T, Y, t, y).mean())
                r["oracle"].append(-orc.logpdf(T, Y, t, y, n_mc=n_mc, seed=s).mean())
                for key, zf in (("model", False), ("model_noflag", True)):
                    mix = predict_mixture(model, x, m, T, Y, t, device=device, zero_flags=zf)
                    r[key].append(-mixture_logpdf(*mix, y)[0].mean().item())
                r["p_anc"].append(orc.prob_ancestor(T, Y))
                r["H"].append(orc.graph_entropy())
            row: dict = {"scm": name, "design": label}
            for k, v in r.items():
                row[k] = float(np.mean(v))
                row[k + "_se"] = float(np.std(v) / np.sqrt(len(v)))
            row["model_minus_oracle"] = row["model"] - row["oracle"]
            rows.append(row)
            print(
                f"{label:22s} truth {row['truth']:.3f} | oracle {row['oracle']:.3f} | "
                f"model {row['model']:.3f} (no-flag {row['model_noflag']:.3f}) | "
                f"gap {row['model_minus_oracle']:+.3f} | P(X{T} anc X{Y}) {row['p_anc']:.2f}"
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Gap to the Bayes oracle on a fixed bank
# --------------------------------------------------------------------------- #
def _item_scores(model, item, device, normalise=True, zero_flags=False):
    """Per query point of a bank item: log p_model(y | do(T = t), context) on
    the RAW scale, the PIT value F_model(y), and exp(entropy) of the weights."""
    mix = predict_mixture(
        model, item["ctx_x"], item["ctx_int"], int(item["T"]), int(item["Y"]),
        item["t"], device=device, zero_flags=zero_flags, normalise=normalise,
    )  # fmt: skip
    logp = mixture_logpdf(*mix, item["y"])[0]
    pit = mixture_cdf(*mix, item["y"])[0]
    eff = effective_components(mix[2])[0]
    return logp.cpu().numpy(), pit.cpu().numpy(), eff.cpu().numpy()


def run_bank_gap(
    model: torch.nn.Module,
    bank: list[dict],
    *,
    device,
    normalise: bool = True,
    flag_ablation: bool = True,
) -> dict[str, float]:
    """Excess NLL of the model over the exact oracle on the bank items
    (`tasks.build_bank`), overall and per label, plus 50% / 90% interval
    coverage, effective mixture components and, with flag_ablation, the
    log-density gained from the intervention flags. Keys are the column names
    of the training-time oracle_gap.csv. The model must be in eval mode."""
    gaps: dict[str, list] = {}
    pits, effs, flag_gain = [], [], []
    for it in bank:
        lp, pit, eff = _item_scores(model, it, device, normalise)
        gap = float(np.mean(it["oracle_logp"] - lp))  # nats per query point
        for lab in it["labels"]:
            gaps.setdefault(lab, []).append(gap)
        pits.append(pit)
        effs.append(eff)
        if flag_ablation and it["n_int_T"] > 0:
            lp0, _, _ = _item_scores(model, it, device, normalise, zero_flags=True)
            flag_gain.append(float(np.mean(lp - lp0)))

    pit = np.concatenate(pits)
    row = {f"oracle_gap/{k}": float(np.mean(v)) for k, v in sorted(gaps.items())}
    row["calib/cover50"] = float(np.mean(np.abs(pit - 0.5) < 0.25))
    row["calib/cover90"] = float(np.mean(np.abs(pit - 0.5) < 0.45))
    row["mog/effective_components"] = float(np.mean(np.concatenate(effs)))
    if flag_ablation:
        row["flag/logp_gain"] = float(np.mean(flag_gain)) if flag_gain else float("nan")
    return row
