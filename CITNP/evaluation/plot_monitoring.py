"""
Plot oracle_gap.csv and train_stats.csv for one or more runs.

    python -m CITNP.evaluation.plot_monitoring \
        /content/drive/MyDrive/alcfm/results/oracle_d3_s_colab \
        /content/drive/MyDrive/alcfm/results/oracle_d3_m_colab \
        --out monitoring.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

GAP_PANELS = [
    ("oracle_gap/all", "excess NLL: all"),
    ("oracle_gap/nT=0", "excess NLL: no do-rows on T"),
    ("oracle_gap/nT=1-10", "excess NLL: 1-10 do-rows on T"),
    ("oracle_gap/nT=11+", "excess NLL: 11+ do-rows on T"),
]
OTHER_PANELS = [
    ("oracle", "calib/cover90", "90% interval coverage (target 0.9)"),
    ("oracle", "flag/logp_gain", "log-density gained from flags"),
    ("oracle", "mog/effective_components", "effective mixture components"),
    ("train", "train/loss_mean", "training loss"),
    ("train", "train/clip_frac", "fraction of clipped steps"),
    ("train", "train/data_wait_frac", "fraction of time waiting for data"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="monitoring.png")
    a = ap.parse_args()

    runs = {}
    for d in a.run_dirs:
        d = Path(d)
        runs[d.name] = {
            "oracle": pd.read_csv(d / "oracle_gap.csv") if (d / "oracle_gap.csv").exists() else None,
            "train": pd.read_csv(d / "train_stats.csv") if (d / "train_stats.csv").exists() else None,
        }

    panels = [("oracle", k, t) for k, t in GAP_PANELS] + OTHER_PANELS
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, (src, key, title) in zip(axes.flat, panels):
        for name, r in runs.items():
            df = r[src]
            if df is not None and key in df:
                ax.plot(df["step"], df[key], label=name)
        if key.startswith("oracle_gap"):
            ax.axhline(0, color="k", lw=0.6)
        if key == "calib/cover90":
            ax.axhline(0.9, color="k", lw=0.6, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
