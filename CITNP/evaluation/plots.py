"""All evaluation figures. Reads saved CSVs (or DataFrames loaded from them)
only; never imports torch or calls a model.

K-shot figures, from <prefix>.csv and <prefix>_by_t.csv written by `kshot`:
    <prefix>_main.png       KL vs number of experiments, 4 curves, oracle dashed
    <prefix>_by_query.png   the same per query
    <prefix>_effect.png     implied-effect error vs number of experiments
    <prefix>_flag.png       model-vs-oracle KL with flags on (solid) and off (dashed)
    <prefix>_varratio.png   model / oracle predictive variance (1 = matches oracle)
    <prefix>_qtype.png      KL to truth split by query type
    <prefix>_by_t.png       KL to truth against |do-value t|, for a few k
Training monitoring, from oracle_gap.csv / train_stats.csv in run directories:
    plot_monitoring(run_dirs, out)

KL = Kullback-Leibler divergence, in nats per query point. Curves are means
over seeds (queries averaged within a seed first), bands are +/- one standard
error over seeds. k = 0 (observational only) is shared by all designs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from CITNP.evaluation.tasks import DESIGNS, MAIN_QUERY, QUERIES, qname

COLORS = {
    "random": "C0",
    "do(treatment)": "C1",
    "do(confounder)": "C2",
    "do(outcome)": "C3",
}
XLAB = "number of experiments (interventional samples)"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def with_k0(df: pd.DataFrame, design: str) -> pd.DataFrame:
    """Rows for one design, with the shared observational-only (k = 0) rows prepended."""
    return pd.concat([df[df.k == 0], df[df.design == design]])


def seed_curve(d: pd.DataFrame, col, agg="mean"):
    """Mean and standard error over seeds of `col` against k."""
    per_seed = d.groupby(["seed", "k"])[col].agg(agg)
    g = per_seed.groupby("k")
    return g.mean(), g.sem()


def _curve(
    df: pd.DataFrame, design: str, cols: list[str], query: str | None = None
):
    """Mean and s.e. over seeds of cols vs k, with k = 0 prepended; averages
    over queries within a seed first when query is None."""
    d = df if query is None else df[df["query"] == query]
    return seed_curve(with_k0(d, design), cols)


def band(ax, mu, se, color, ls="-", label=None, marker="o") -> None:
    ax.plot(
        mu.index, mu.values, ls, marker=marker, ms=3, color=color, label=label
    )
    ax.fill_between(mu.index, mu - se, mu + se, color=color, alpha=0.15)


def finish(ax, ks, title: str, ylabel: str | None = None) -> None:
    ax.set_xticks(ks)
    ax.set_xlabel(XLAB)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def _save(fig, path: str, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# K-shot: main figures
# --------------------------------------------------------------------------- #
def plot_kshot_main(df: pd.DataFrame, prefix: str) -> None:
    """KL to truth (main query, mean of queries) and to the oracle vs k."""
    panels = [
        (
            "kl_truth_model",
            "kl_truth_oracle",
            MAIN_QUERY,
            f"KL(truth || ·): {MAIN_QUERY}",
        ),
        (
            "kl_truth_model",
            "kl_truth_oracle",
            None,
            "KL(truth || ·): mean of 6 queries",
        ),
        (
            "kl_oracle_model",
            None,
            None,
            "KL(Bayes oracle || model): mean of 6 queries",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (mcol, ocol, q, title) in zip(axes, panels):
        for d in DESIGNS:
            cols = [mcol] + ([ocol] if ocol else [])
            mu, se = _curve(df, d, cols, q)
            ax.plot(
                mu.index,
                mu[mcol],
                "-o",
                ms=3,
                color=COLORS[d],
                label=f"model, {d}",
            )
            ax.fill_between(mu.index, mu[mcol] - se[mcol], mu[mcol] + se[mcol],
                            color=COLORS[d], alpha=0.15)  # fmt: skip
            if ocol:
                ax.plot(
                    mu.index,
                    mu[ocol],
                    "--",
                    color=COLORS[d],
                    label=f"oracle, {d}",
                )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(sorted(df.k.unique()))
        ax.set_xlabel(XLAB)
        ax.set_title(title)
    axes[0].set_ylabel("KL, nats per query point")
    axes[0].legend(fontsize=7, ncol=2)
    _save(fig, prefix + "_main.png")


def plot_kshot_by_query(df: pd.DataFrame, prefix: str) -> None:
    """KL to truth per query, model solid, oracle dashed."""
    queries = [qname(T, Y) for T, Y in QUERIES]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for ax, q in zip(axes.flat, queries):
        for d in DESIGNS:
            mu, _ = _curve(df, d, ["kl_truth_model", "kl_truth_oracle"], q)
            ax.plot(
                mu.index,
                mu.kl_truth_model,
                "-o",
                ms=3,
                color=COLORS[d],
                label=f"model, {d}",
            )
            ax.plot(
                mu.index,
                mu.kl_truth_oracle,
                "--",
                color=COLORS[d],
                label=f"oracle, {d}",
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"KL(truth || ·): {q}")
    for ax in axes[1]:
        ax.set_xlabel(XLAB)
    axes[0, 0].legend(fontsize=7, ncol=2)
    _save(fig, prefix + "_by_query.png")


def plot_kshot_effect(df: pd.DataFrame, prefix: str) -> None:
    """Implied causal effect error per query (model solid, oracle dashed,
    naive observational regression dotted)."""
    queries = [qname(T, Y) for T, Y in QUERIES]
    cols = [
        "slope_err_model_truth",
        "slope_err_oracle_truth",
        "slope_err_naive_truth",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for ax, q in zip(axes.flat, queries):
        for d in DESIGNS:
            mu, _ = _curve(df, d, cols, q)
            ax.plot(mu.index, mu.slope_err_model_truth, "-o", ms=3, color=COLORS[d],
                    label=f"model, {d}")  # fmt: skip
            ax.plot(mu.index, mu.slope_err_oracle_truth, "--", color=COLORS[d],
                    label=f"oracle, {d}")  # fmt: skip
        ax.axhline(mu.slope_err_naive_truth.iloc[0], color="r", ls=":", lw=1,
                   label="naive obs regression")  # fmt: skip
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"effect error (estimate - truth): {q}")
    for ax in axes[1]:
        ax.set_xlabel(XLAB)
    axes[0, 0].legend(fontsize=7, ncol=2)
    _save(fig, prefix + "_effect.png")


# --------------------------------------------------------------------------- #
# K-shot: diagnostics
# --------------------------------------------------------------------------- #
def plot_flag(df: pd.DataFrame, prefix: str, ks) -> None:
    """KL(oracle || model) with intervention flags on and off."""
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
    fig.suptitle(
        "Does the model use the intervention flags? "
        "(gap between the curves = flag contribution)"
    )
    _save(fig, prefix + "_flag.png")


def plot_varratio(df: pd.DataFrame, prefix: str, ks) -> None:
    """Model / oracle predictive variance against k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in DESIGNS:
        # median over queries within a seed: ratios are skewed
        mu, se = seed_curve(
            with_k0(df, d), "var_ratio_model_oracle", agg="median"
        )
        band(ax, mu, se, COLORS[d], label=d)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    finish(ax, ks, "Predictive variance, model / oracle (>1 underconfident, <1 overconfident)",
           "variance ratio (log scale)")  # fmt: skip
    ax.legend(fontsize=8)
    _save(fig, prefix + "_varratio.png")


def plot_qtype(
    df: pd.DataFrame, prefix: str, ks, *, min_contexts: int = 20
) -> None:
    """KL to truth split by query type. Points backed by fewer than
    `min_contexts` contexts (seeds) are skipped; each point is annotated with
    its number of contexts."""
    types = [
        t
        for t in ["seen", "unseen_identified", "unidentified"]
        if t in set(df.query_type)
    ]
    fig, axes = plt.subplots(
        1, len(types), figsize=(6 * len(types), 4.5), sharey=True
    )
    axes = [axes] if len(types) == 1 else axes
    for ax, qt in zip(axes, types):
        for d in DESIGNS:
            h = with_k0(df, d)
            h = h[h.query_type == qt]
            if h.empty:
                continue
            n = h.groupby("k")["seed"].nunique()
            keep = n.index[n >= min_contexts]
            if len(keep) == 0:
                continue
            mu, se = seed_curve(h, "kl_truth_model")
            mu, se, n = mu.loc[keep], se.loc[keep], n.loc[keep]
            band(ax, mu, se, COLORS[d], "-", f"model, {d}")
            for k, v, c in zip(mu.index, mu.values, n.values):
                ax.annotate(str(c), (k, v), textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=6, color=COLORS[d])  # fmt: skip
            mu, _ = seed_curve(h, "kl_truth_oracle")
            mu = mu.loc[keep]
            ax.plot(
                mu.index, mu.values, "--", color=COLORS[d], label=f"oracle, {d}"
            )
        ax.axhline(0, color="k", lw=0.5)
        finish(ax, ks, f"KL(truth || ·): query type = {qt}")
    axes[0].set_ylabel("KL, nats per query point")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        f"numbers = contexts per point; points with < {min_contexts} contexts omitted"
    )
    _save(fig, prefix + "_qtype.png")


def plot_by_t(bt: pd.DataFrame, prefix: str, ks) -> None:
    """KL to truth against |do-value t| for k in {0, 10, max k}."""
    show = [k for k in (0, 10, max(ks)) if k in ks]
    bt = bt.assign(abs_t=bt.t.abs().round(6))
    fig, axes = plt.subplots(1, len(DESIGNS), figsize=(20, 4.5), sharey=True)
    for ax, d in zip(axes, DESIGNS):
        h = with_k0(bt, d)
        for i, k in enumerate(show):
            hk = h[h.k == k]
            per_seed = hk.groupby(["seed", "abs_t"])[
                ["kl_truth_model", "kl_truth_oracle"]
            ].mean()
            g = per_seed.groupby("abs_t").mean()
            c = plt.cm.viridis(i / max(len(show) - 1, 1))
            ax.plot(
                g.index,
                g.kl_truth_model,
                "-o",
                ms=3,
                color=c,
                label=f"model, k={k}",
            )
            ax.plot(
                g.index, g.kl_truth_oracle, "--", color=c, label=f"oracle, k={k}"
            )
        ax.set_xlabel("|do-value t| (context do-values have sd 1.5)")
        ax.set_title(f"KL(truth || ·) vs |t|: {d}")
    axes[0].set_ylabel("KL, nats per query point")
    axes[0].legend(fontsize=7)
    _save(fig, prefix + "_by_t.png")


def plot_kshot_all(csv_path: str) -> None:
    """Every k-shot figure from <prefix>.csv and, if present, <prefix>_by_t.csv."""
    prefix = csv_path.removesuffix(".csv")
    df = pd.read_csv(csv_path)
    ks = sorted(df.k.unique())
    plot_kshot_main(df, prefix)
    plot_kshot_by_query(df, prefix)
    plot_kshot_effect(df, prefix)
    plot_flag(df, prefix, ks)
    plot_varratio(df, prefix, ks)
    plot_qtype(df, prefix, ks)
    written = ["_main", "_by_query", "_effect", "_flag", "_varratio", "_qtype"]
    if Path(prefix + "_by_t.csv").exists():
        plot_by_t(pd.read_csv(prefix + "_by_t.csv"), prefix, ks)
        written.append("_by_t")
    else:
        print(f"no {prefix}_by_t.csv, skipping the |t| figure")
    print("wrote " + ", ".join(f"{prefix}{s}.png" for s in written))


# --------------------------------------------------------------------------- #
# Training monitoring
# --------------------------------------------------------------------------- #
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


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def plot_monitoring(run_dirs: list[str], out: str = "monitoring.png") -> None:
    """oracle_gap.csv and train_stats.csv of one or more runs against step."""
    runs = {}
    for d in map(Path, run_dirs):
        runs[d.name] = {
            "oracle": _read_if_exists(d / "oracle_gap.csv"),
            "train": _read_if_exists(d / "train_stats.csv"),
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
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    _save(fig, out, dpi=120)
    print(f"wrote {out}")
