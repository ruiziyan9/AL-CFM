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

import pandas as pd
import torch

from CITNP.evaluation.tasks import MAIN_QUERY


def plot_results(df: pd.DataFrame, prefix: str) -> None:
    """The three main k-shot figures (see CITNP.evaluation.plots)."""
    import matplotlib

    matplotlib.use("Agg")
    from CITNP.evaluation import plots

    plots.plot_kshot_main(df, prefix)
    plots.plot_kshot_by_query(df, prefix)
    plots.plot_kshot_effect(df, prefix)
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
    from CITNP.evaluation.experiments import run_kshot
    from CITNP.evaluation.model_api import load_model

    data_cfg = load_config(a.data_config)
    model = load_model(a.ckpt, a.model_config, a.device)
    df, df_t = run_kshot(
        model, data_cfg, device=a.device, k_list=k_list, n_seeds=a.n_seeds,
        n_obs=a.n_obs, random_weights=a.random_weights, do_scale=a.do_scale,
        t_max=a.t_max, n_t=a.n_t, n_y=a.n_y, n_mc=a.n_mc,
        min_graph_prob=a.min_graph_prob,
    )  # fmt: skip
    df.to_csv(a.out, index=False)
    prefix = a.out.removesuffix(".csv")
    df_t.to_csv(prefix + "_by_t.csv", index=False)
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
