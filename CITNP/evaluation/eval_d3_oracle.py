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

import numpy as np
import torch

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.evaluation.model_api import load_model, mixture_logpdf, predict_mixture
from CITNP.evaluation.oracle import LinGaussOracle
from CITNP.evaluation.tasks import (
    context_designs,
    empirical_graph_prior,
    make_context,
    scm_suite,
)

D = 3


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
    for name, scm, (T, Y), why in scm_suite():
        slope, var = scm.do_dist(T, Y)
        print(f"\n=== {name}: Y=X{Y} | do(X{T}=t) ~ N({slope:.2f} t, {var:.3f})  [{why}]")
        for label, regimes in context_designs():
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
