"""
Prior diagnostics for the mixed-regime generator: samples tasks and saves one
PNG grid of histograms.

    python -m CITNP.datasets.plot_mixed_prior \
        --config CITNP/datasets/configs/mixed_linear_gaussian.yaml \
        --num_tasks 2000 --out mixed_prior.png
"""

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import (
    QRY_TYPES,
    SOURCE_ADHOC,
    generate_task,
    make_key,
)


def collect(cfg, num_tasks: int, base_seed: int) -> dict:
    s: dict[str, list] = {
        k: []
        for k in [
            "D",
            "K",
            "has_obs",
            "n_r",
            "imbalance",
            "frac_int",
            "do_values",
            "p01",
            "p99",
            "resampled",
            "max_abs",
            "graph_type",
        ]
    }
    qry_types: Counter = Counter()
    for i in trange(num_tasks):
        task = generate_task(cfg, make_key(SOURCE_ADHOC, 0, base_seed, i))
        m = task.meta
        sizes = np.array(m["regime_sizes"])
        s["D"].append(m["D"])
        s["K"].append(m["K"])
        s["has_obs"].append(int(m["has_obs"]))
        s["n_r"].extend(sizes.tolist())
        s["imbalance"].append(sizes.max() / sizes.min())
        s["frac_int"].append(task.ctx_int.any(1).mean())
        s["do_values"].extend(task.ctx_x[task.ctx_int].tolist())
        s["do_values"].extend(task.qry_x[task.qry_int].tolist())
        x = np.concatenate([task.ctx_x, task.qry_x])
        s["p01"].extend(np.percentile(x, 1, axis=0).tolist())
        s["p99"].extend(np.percentile(x, 99, axis=0).tolist())
        s["resampled"].append(m["resample_attempts"])
        s["max_abs"].append(m["max_abs"])
        s["graph_type"].append(m["graph_type"])
        qry_types.update(m["qry_types"])
    s["qry_types"] = [qry_types[t] for t in QRY_TYPES]
    return s


def plot(s: dict, out: str, title: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    ax = axes.ravel()

    def discrete(a, values, label):
        counts = [np.sum(np.asarray(values) == v) for v in sorted(set(values))]
        a.bar([str(v) for v in sorted(set(values))], counts, color="#4C72B0")
        a.set_title(label)

    discrete(ax[0], s["D"], "D (num variables)")
    discrete(ax[1], s["K"], "K (interventional regimes)")
    discrete(ax[2], s["has_obs"], "has_obs")
    ax[3].hist(s["n_r"], bins=50, color="#4C72B0")
    ax[3].set_yscale("log")
    ax[3].set_title("regime size n_r")
    ax[4].hist(np.log10(s["imbalance"]), bins=50, color="#4C72B0")
    ax[4].set_title("log10(max n_r / min n_r)")
    ax[5].hist(s["frac_int"], bins=40, color="#4C72B0")
    ax[5].set_title("fraction of context rows interventional")
    ax[6].bar(list(QRY_TYPES), s["qry_types"], color="#4C72B0")
    ax[6].set_title("query target types")
    ax[7].hist(s["do_values"], bins=100, color="#4C72B0")
    ax[7].set_title("do-values")
    ax[8].hist(s["p01"], bins=60, alpha=0.6, label="1st pct", color="#4C72B0")
    ax[8].hist(s["p99"], bins=60, alpha=0.6, label="99th pct", color="#DD8452")
    ax[8].set_title("per-variable value range")
    ax[8].legend()
    ax[9].hist(np.log10(s["max_abs"]), bins=60, color="#4C72B0")
    ax[9].set_title("log10(max |value|) per task")
    discrete(ax[10], s["graph_type"], "graph type")
    ax[11].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_tasks", type=int, default=2000)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--out", default="mixed_prior.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s = collect(cfg, args.num_tasks, args.base_seed)
    resampled = np.array(s["resampled"])
    print(
        f"Resample (non-finite or > max_abs_value) rate: {np.mean(resampled > 0):.4f} of tasks, "
        f"{resampled.sum()} extra attempts in total"
    )
    plot(s, args.out, f"{cfg.function_generator}, {args.num_tasks} tasks")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
