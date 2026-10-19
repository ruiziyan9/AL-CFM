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

import torch

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.evaluation.experiments import run_scm_suite
from CITNP.evaluation.model_api import load_model


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
    df = run_scm_suite(
        model, load_config(a.data_config), device=a.device, n_seeds=a.n_seeds,
        n_query=a.n_query, do_scale=a.do_scale, n_mc=a.n_mc,
    )  # fmt: skip
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
