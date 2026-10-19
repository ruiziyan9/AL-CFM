"""
Evaluate a trained mixed-context model on known 3-variable linear-Gaussian
SCMs, against (a) the true interventional distribution and (b) the exact
Bayes-optimal posterior predictive (LinGaussOracle).

    python -m CITNP.evaluation.eval_d3_oracle \
        --ckpt <run_dir>/best_model.pt \
        --model_config CITNP/experiments/configs/smoke_d3_model.yaml \
        --data_config  CITNP/experiments/configs/oracle_d3.yaml \
        --out results_d3_oracle.csv

All NLLs are per query row, in nats, on the RAW data scale (model predictions
are mapped back from the per-task standardised scale with the Jacobian).

The oracle is only Bayes-optimal for the prior the model was trained on
(`linearnonidentifiable`, normalise: false). For a model trained on another
prior, the "truth" columns are still exact; the oracle columns are then just
a reference.
"""

import argparse
import csv
import itertools

import numpy as np
import torch

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import sample_graph
from CITNP.evaluation.model_api import load_model, mixture_logpdf, predict_mixture
from CITNP.evaluation.oracle import LinearSCM, LinGaussOracle, dag_key
from CITNP.evaluation.oracle_lingauss import make_context

D = 3


# --------------------------------------------------------------------------- #
# Graph prior of the training generator (incl. its random variable permutation)
# --------------------------------------------------------------------------- #
def empirical_graph_prior(cfg, n=20000, seed=0):
    np.random.seed(seed)
    import random

    random.seed(seed)
    counts = {}
    for _ in range(n):
        g, _ = sample_graph(cfg, D)
        p = np.random.permutation(D)
        g = g[np.ix_(p, p)].astype(int)
        counts[dag_key(g)] = counts.get(dag_key(g), 0) + 1
    return {k: np.log(v / n) for k, v in counts.items()}


# --------------------------------------------------------------------------- #
# Known-truth SCMs. Each: (name, SCM, query (T, Y), why it is interesting)
# --------------------------------------------------------------------------- #
def scms():
    out = []
    B = np.zeros((D, D)); B[0, 1] = 0.9; B[1, 2] = -0.8
    out.append(("chain_0to1to2", LinearSCM(B, np.array([0.3, 0.2, 0.25])), (0, 2),
                "obs MEC = {chain, reverse chain, fork}: effect not identified"))
    B = np.zeros((D, D)); B[1, 0] = 0.9; B[1, 2] = 0.8
    out.append(("fork_1_to_0_2", LinearSCM(B, np.array([0.2, 0.3, 0.25])), (0, 2),
                "same MEC, true effect is ZERO (confounded association)"))
    B = np.zeros((D, D)); B[0, 2] = 0.9; B[1, 2] = 0.7
    out.append(("collider_0_1_to_2", LinearSCM(B, np.array([0.3, 0.3, 0.2])), (0, 2),
                "v-structure: identified from obs, experiments add little"))
    return out


def designs(n_obs=128, m_grid=(0, 8, 16, 32, 64, 128)):
    """(label, regimes). Total rows stay inside training n_ctx_range."""
    out = [("obs_only", [(None, n_obs)])]
    for node, m in itertools.product(range(D), m_grid[1:]):
        out.append((f"obs+do(X{node})x{m}", [(None, n_obs), (node, m)]))
    out.append(("do(X0)_only_x128", [(0, 128)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--data_config", required=True)
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--n_query", type=int, default=256)
    ap.add_argument("--do_scale", type=float, default=1.5)
    ap.add_argument("--n_mc", type=int, default=1000)
    ap.add_argument("--out", default="results_d3_oracle.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model = load_model(a.ckpt, a.model_config, a.device)
    prior = empirical_graph_prior(load_config(a.data_config))

    rows = []
    for name, scm, (T, Y), why in scms():
        slope, var = scm.do_dist(T, Y)
        print(f"\n=== {name}: Y=X{Y} | do(X{T}=t) ~ N({slope:.2f} t, {var:.3f})  [{why}]")
        for label, regimes in designs():
            r = {k: [] for k in ("truth", "oracle", "model", "model_noflag", "p_anc", "H")}
            for s in range(a.n_seeds):
                rng = np.random.default_rng(1000 * s + 7)
                x, m = make_context(scm, regimes, a.do_scale, rng)
                t = rng.normal(0, a.do_scale, a.n_query)
                xq, _ = scm.sample(a.n_query, T, t, rng)
                y = xq[:, Y]
                orc = LinGaussOracle(D, prior).fit(x, m)
                r["truth"].append(-scm.do_logpdf(T, Y, t, y).mean())
                r["oracle"].append(-orc.logpdf(T, Y, t, y, n_mc=a.n_mc, seed=s).mean())
                for key, zf in (("model", False), ("model_noflag", True)):
                    mix = predict_mixture(model, x, m, T, Y, t, device=a.device, zero_flags=zf)
                    r[key].append(-mixture_logpdf(*mix, y)[0].mean().item())
                r["p_anc"].append(orc.prob_ancestor(T, Y))
                r["H"].append(orc.graph_entropy())
            row = {"scm": name, "design": label}
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

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
