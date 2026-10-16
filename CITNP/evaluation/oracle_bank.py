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
import time
from collections import Counter

import numpy as np

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_ADHOC,
    generate_task,
    make_key,
)
from CITNP.evaluation.eval_d3_oracle import empirical_graph_prior
from CITNP.evaluation.oracle_lingauss import LinGaussOracle

BUCKETS = ("nT=0", "nT=1-10", "nT=11+")


def n_int_bucket(n: int) -> str:
    """Bucket of the number of context rows intervened on the query treatment."""
    if n == 0:
        return BUCKETS[0]
    return BUCKETS[1] if n <= 10 else BUCKETS[2]


def build_bank(cfg, per_bucket=150, D=3, base_seed=4242, n_mc=2000, max_tasks=100_000):
    """Return a list of question dicts; `per_bucket` questions (T) per bucket,
    each expanded over all D - 1 outcomes Y != T."""
    if cfg.function_generator != "linearnonidentifiable" or cfg.normalise:
        raise ValueError("oracle bank needs linearnonidentifiable with normalise: false")
    prior = empirical_graph_prior(cfg)
    counts = {b: 0 for b in BUCKETS}
    items = []
    t0 = time.time()
    for i in range(max_tasks):
        if all(c >= per_bucket for c in counts.values()):
            break
        # SOURCE_ADHOC keys never collide with training / val / test streams.
        task = generate_task(cfg, make_key(SOURCE_ADHOC, 0, base_seed, i), num_variables=D)
        ctx_x = task.ctx_x.astype(np.float64)
        ctx_int = task.ctx_int.astype(bool)
        qx = task.qry_x.astype(np.float64)
        qi = task.qry_int.astype(bool)
        # Grouped mode: each query row intervenes on exactly one variable.
        treatments = sorted({int(np.argmax(r)) for r in qi})

        todo = []
        for T in treatments:
            n_T = int(ctx_int[:, T].sum())
            b = n_int_bucket(n_T)
            if counts[b] < per_bucket:
                todo.append((T, b, n_T))
        if not todo:
            continue

        orc = LinGaussOracle(D, prior).fit(ctx_x, ctx_int)
        K = int(task.meta["K"])
        has_obs = bool(task.meta["has_obs"])
        for T, b, n_T in todo:
            rows = qi[:, T]
            t = qx[rows, T]
            for Y in range(D):
                if Y == T:
                    continue
                y = qx[rows, Y]
                items.append(
                    dict(
                        ctx_x=task.ctx_x.astype(np.float32),
                        ctx_int=ctx_int,
                        T=T,
                        Y=Y,
                        t=t.astype(np.float32),
                        y=y.astype(np.float32),
                        oracle_logp=orc.logpdf(T, Y, t, y, n_mc=n_mc, seed=i),
                        n_int_T=n_T,
                        labels=[
                            "all",
                            b,
                            f"K={'2+' if K >= 2 else K}",
                            f"has_obs={has_obs}",
                        ],
                    )
                )
            counts[b] += 1
        if i % 200 == 0:
            print(f"task {i}: {counts}  ({time.time() - t0:.0f}s)")
    else:
        print(f"WARNING: hit max_tasks={max_tasks} before filling quotas: {counts}")
    return items


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
