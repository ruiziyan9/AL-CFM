"""
Extra diagnostic figures from the CSVs written by CITNP.evaluation.eval_d3_kshot.

    python plot_kshot_extra.py kshot_m_best_k50.csv

Reads <name>.csv and <name>_by_t.csv and writes, next to them:
    <name>_flag.png       model-vs-oracle KL with flags on (solid) and off (dashed)
    <name>_varratio.png   model / oracle predictive variance (1 = matches oracle)
    <name>_qtype.png      KL to truth split by query type (seen / unseen_identified / unidentified)
    <name>_by_t.png       KL to truth against |do-value t|, for a few k

KL = Kullback-Leibler divergence, in nats per query point.
Curves are means over seeds (queries averaged within a seed first), bands are
+/- one standard error over seeds. k = 0 (observational only) is shared by all designs.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DESIGNS = ["random", "do(treatment)", "do(confounder)", "do(outcome)"]
COLORS = {"random": "C0", "do(treatment)": "C1", "do(confounder)": "C2", "do(outcome)": "C3"}
XLAB = "number of experiments (interventional samples)"


def with_k0(df, design):
    """Rows for one design, with the shared observational-only (k = 0) rows prepended."""
    return pd.concat([df[df.k == 0], df[df.design == design]])


def seed_curve(d, col, agg="mean"):
    """Mean and standard error over seeds of `col` against k."""
    per_seed = d.groupby(["seed", "k"])[col].agg(agg)
    g = per_seed.groupby("k")
    return g.mean(), g.sem()


def band(ax, mu, se, color, ls="-", label=None, marker="o"):
    ax.plot(mu.index, mu.values, ls, marker=marker, ms=3, color=color, label=label)
    ax.fill_between(mu.index, mu - se, mu + se, color=color, alpha=0.15)


def finish(ax, ks, title, ylabel=None):
    ax.set_xticks(ks)
    ax.set_xlabel(XLAB)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_flag(df, prefix, ks):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for ax, d in zip(axes, DESIGNS):
        h = with_k0(df, d)
        mu, se = seed_curve(h, "kl_oracle_model")
        band(ax, mu, se, COLORS[d], "-", "flags on")
        mu, se = seed_curve(h, "kl_oracle_model_noflag")
        band(ax, mu, se, "grey", "--", "flags off")
        ax.axhline(0, color="k", lw=0.5)
        finish(ax, ks, f"KL(oracle || model): {d}")
    axes[0].set_ylabel("KL, nats per query point")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does the model use the intervention flags? (gap between the curves = flag contribution)")
    fig.tight_layout()
    fig.savefig(prefix + "_flag.png", dpi=150)
    plt.close(fig)


def plot_varratio(df, prefix, ks):
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in DESIGNS:
        # median over queries within a seed: ratios are skewed
        mu, se = seed_curve(with_k0(df, d), "var_ratio_model_oracle", agg="median")
        band(ax, mu, se, COLORS[d], label=d)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    finish(ax, ks, "Predictive variance, model / oracle (>1 underconfident, <1 overconfident)",
           "variance ratio (log scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(prefix + "_varratio.png", dpi=150)
    plt.close(fig)


def plot_qtype(df, prefix, ks):
    types = [t for t in ["seen", "unseen_identified", "unidentified"] if t in set(df.query_type)]
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 4.5), sharey=True)
    axes = [axes] if len(types) == 1 else axes
    for ax, qt in zip(axes, types):
        for d in DESIGNS:
            h = with_k0(df, d)
            h = h[h.query_type == qt]
            if h.empty:
                continue
            mu, se = seed_curve(h, "kl_truth_model")
            band(ax, mu, se, COLORS[d], "-", f"model, {d}")
            mu, _ = seed_curve(h, "kl_truth_oracle")
            ax.plot(mu.index, mu.values, "--", color=COLORS[d], label=f"oracle, {d}")
        ax.axhline(0, color="k", lw=0.5)
        finish(ax, ks, f"KL(truth || ·): query type = {qt}")
    axes[0].set_ylabel("KL, nats per query point")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(prefix + "_qtype.png", dpi=150)
    plt.close(fig)


def plot_by_t(bt, prefix, ks):
    show = [k for k in (0, 10, max(ks)) if k in ks]
    bt = bt.assign(abs_t=bt.t.abs().round(6))
    fig, axes = plt.subplots(1, len(DESIGNS), figsize=(20, 4.5), sharey=True)
    for ax, d in zip(axes, DESIGNS):
        h = with_k0(bt, d)
        for i, k in enumerate(show):
            hk = h[h.k == k]
            per_seed = hk.groupby(["seed", "abs_t"])[["kl_truth_model", "kl_truth_oracle"]].mean()
            g = per_seed.groupby("abs_t").mean()
            c = plt.cm.viridis(i / max(len(show) - 1, 1))
            ax.plot(g.index, g.kl_truth_model, "-o", ms=3, color=c, label=f"model, k={k}")
            ax.plot(g.index, g.kl_truth_oracle, "--", color=c, label=f"oracle, k={k}")
        ax.set_xlabel("|do-value t| (context do-values have sd 1.5)")
        ax.set_title(f"KL(truth || ·) vs |t|: {d}")
    axes[0].set_ylabel("KL, nats per query point")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(prefix + "_by_t.png", dpi=150)
    plt.close(fig)


def main():
    path = sys.argv[1]
    prefix = path.removesuffix(".csv")
    df = pd.read_csv(path)
    ks = sorted(df.k.unique())
    plot_flag(df, prefix, ks)
    plot_varratio(df, prefix, ks)
    plot_qtype(df, prefix, ks)
    try:
        bt = pd.read_csv(prefix + "_by_t.csv")
        plot_by_t(bt, prefix, ks)
    except FileNotFoundError:
        print(f"no {prefix}_by_t.csv, skipping the |t| figure")
    print(f"wrote {prefix}_flag.png, _varratio.png, _qtype.png, _by_t.png")


if __name__ == "__main__":
    main()
