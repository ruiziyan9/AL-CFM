"""
Fixed validation bank for tracking a model's gap to the exact Bayes oracle.

Build it once (CPU is fine, a few minutes for D=3):

    python -m CITNP.evaluation.oracle_bank \
        --data_config CITNP/experiments/configs/oracle_d3.yaml \
        --out /content/drive/MyDrive/alcfm/oracle_bank_d3.pkl

Each item is one question (task, treatment T, outcome Y): the raw context,
the query values (t, y), the oracle log-density of every query point, and
slice labels. Tasks come from the training prior. Rejection sampling fills a
quota per bucket of "context rows intervened on T" (0, 1-10, 11+), so the rare
few-shot regime is well represented. Conditioning on a bucket keeps the oracle
exact, since it is Bayes-optimal for every context.

Only valid for the conjugate prior the oracle assumes
(function_generator: linearnonidentifiable, normalise: false).
"""

import argparse
import pickle
from collections import Counter

import numpy as np

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.evaluation.tasks import BUCKETS, build_bank, n_int_bucket  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_bucket", type=int, default=150)
    ap.add_argument("--n_mc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()

    cfg = load_config(a.data_config)
    items = build_bank(cfg, per_bucket=a.per_bucket, base_seed=a.seed, n_mc=a.n_mc)
    with open(a.out, "wb") as f:
        pickle.dump({"items": items, "data_config": a.data_config, "seed": a.seed}, f)

    labels = Counter(lab for it in items for lab in it["labels"])
    print(f"\nwrote {len(items)} questions to {a.out}")
    for lab, n in sorted(labels.items()):
        nll = np.mean([-it["oracle_logp"].mean() for it in items if lab in it["labels"]])
        print(f"  {lab:14s} n={n:4d}  oracle NLL {nll:.4f}")


if __name__ == "__main__":
    main()
