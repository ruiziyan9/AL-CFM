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
K-shot figures on tasks drawn from the training prior, from the CSV part(s)
written by `kshot-prior` (ONLY these show a true oracle lower bound; per D):
    <prefix>_D{D}_main.png         (a) KL to truth, model solid / oracle dashed,
                                   (b) excess KL(truth) with a line at 0,
                                   (c) KL(oracle || model); curves = mean +/- s.e.
                                   over tasks, pairs averaged within a task first
    <prefix>_D{D}_by_relation.png  excess KL by question class (the bound need
                                   not hold within a class)
Training monitoring, from oracle_gap.csv / train_stats.csv in run directories:
    plot_monitoring(run_dirs, out)
Flag probe, from <prefix>.csv written by `probe-flag`:
    <prefix>.png            probe AUC (flags on / off) and relative change of the
                            representation against encoder depth
Intervention-encoding diagnostics, from labelled result prefixes P of
`probe-intervention` / `context-edits` (`plot_intervention_diagnostics`;
PNG and PDF in an output directory; see
CITNP/evaluation/INTERVENTION_DIAGNOSTICS_README.md):
    probes_by_layer            probe balanced accuracy against layer
    edit_shift_scatter_*       model vs oracle shift per row (mean shift, KL)
    edit_shift_curve           shift against the number of edited rows
    edit_sensitivity_ratio     model / oracle ratio by stratum (+ .csv)
    normalisation_effect       model shift, original vs edited normalisation
    secondary_discrepancy      KL(oracle || model), unedited vs edited

KL = Kullback-Leibler divergence, in nats per query point. Curves are means
over seeds (queries averaged within a seed first), bands are +/- one standard
error over seeds. k = 0 (observational only) is shared by all designs.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def seed_curve(d: pd.DataFrame, col, agg="mean", unit="seed"):
    """Mean and standard error over seeds (or, with unit="task", over tasks)
    of `col` against k."""
    per_seed = d.groupby([unit, "k"])[col].agg(agg)
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
# K-shot on tasks drawn from the training prior
# --------------------------------------------------------------------------- #
PRIOR_COLORS = {
    "random": "C0",
    "do(T)": "C1",
    "do(Y)": "C3",
    "do(other)": "C2",
}
_ID_COLS = ["D", "task", "T", "Y", "design", "k"]


def _prior_curve(df: pd.DataFrame, design: str, cols: list[str]):
    """Mean and s.e. over tasks of cols vs k, with k = 0 prepended; pairs are
    averaged within a task first."""
    return seed_curve(with_k0(df, design), cols, unit="task")


def plot_kshot_prior_main(df: pd.DataFrame, prefix: str) -> None:
    """One figure <prefix>_D{D}_main.png per D, from kshot-prior rows.

    (a) KL(truth || model) solid and KL(truth || oracle) dashed per design;
    (b) excess KL(truth) = KL(truth || model) - KL(truth || oracle), +/- s.e.
    over tasks (>= 0 in expectation: the oracle is a lower bound);
    (c) KL(oracle || model), which equals (b) in expectation.
    """
    for D, dd in df.groupby("D"):
        ks = sorted(dd.k.unique())
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for d in PRIOR_COLORS:
            c = PRIOR_COLORS[d]
            mu, se = _prior_curve(
                dd, d, ["kl_truth_model", "kl_truth_oracle", "excess_kl_truth",
                        "kl_oracle_model"]
            )  # fmt: skip
            band(axes[0], mu["kl_truth_model"], se["kl_truth_model"], c,
                 label=f"model, {d}")  # fmt: skip
            axes[0].plot(mu.index, mu["kl_truth_oracle"], "--", color=c,
                         label=f"oracle, {d}")  # fmt: skip
            band(axes[1], mu["excess_kl_truth"], se["excess_kl_truth"], c,
                 label=d)  # fmt: skip
            band(axes[2], mu["kl_oracle_model"], se["kl_oracle_model"], c,
                 label=d)  # fmt: skip
        titles = [
            f"D={D}: KL(truth || ·)",
            f"D={D}: excess KL(truth) = KL(truth||model) - KL(truth||oracle)",
            f"D={D}: KL(oracle || model)",
        ]
        for ax, title in zip(axes, titles):
            ax.axhline(0, color="k", lw=0.5)
            finish(ax, ks, title)
        axes[0].set_ylabel("KL, nats per query point")
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].legend(fontsize=7)
        _save(fig, f"{prefix}_D{D}_main.png")


def plot_kshot_prior_by_relation(df: pd.DataFrame, prefix: str) -> None:
    """One figure <prefix>_D{D}_by_relation.png per D: excess KL(truth) by
    design, one panel per question class. The lower bound holds on average
    over the whole prior, not within a class, so a facet may dip below 0."""
    for D, dd in df.groupby("D"):
        rels = sorted(dd["relation"].unique())
        ks = sorted(dd.k.unique())
        fig, axes = plt.subplots(
            1, len(rels), figsize=(4.5 * len(rels), 4.5), sharey=True,
            squeeze=False,
        )  # fmt: skip
        for ax, rel in zip(axes[0], rels):
            sub = dd[dd["relation"] == rel]
            for d, c in PRIOR_COLORS.items():
                mu, se = _prior_curve(sub, d, ["excess_kl_truth"])
                band(ax, mu["excess_kl_truth"], se["excess_kl_truth"], c,
                     label=d)  # fmt: skip
            ax.axhline(0, color="k", lw=0.5)
            finish(ax, ks, rel)
        axes[0][0].set_ylabel("excess KL(truth), nats per query point")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(
            f"D={D}: excess KL by question class (the bound >= 0 need not "
            "hold within a class)"
        )
        _save(fig, f"{prefix}_D{D}_by_relation.png")


def plot_kshot_prior_all(
    csv_paths: list[str], prefix: str | None = None
) -> None:
    """Every kshot-prior figure from one or more CSV parts (array jobs over
    task_offset), concatenated. Raises if a (D, task, T, Y, design, k) row
    appears twice, i.e. parts overlap."""
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    if df.duplicated(_ID_COLS).any():
        raise ValueError("overlapping parts: duplicate rows")
    if prefix is None:
        prefix = re.sub(r"_part\d+$", "", csv_paths[0].removesuffix(".csv"))
    plot_kshot_prior_main(df, prefix)
    plot_kshot_prior_by_relation(df, prefix)
    written = [
        f"{prefix}_D{D}{s}.png"
        for D in sorted(df["D"].unique())
        for s in ("_main", "_by_relation")
    ]
    print("wrote " + ", ".join(written))


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


# --------------------------------------------------------------------------- #
# Flag probe
# --------------------------------------------------------------------------- #
def plot_flag_probe(df: pd.DataFrame, out: str) -> None:
    """Two panels against encoder depth (`run_flag_probe` output):
    left, out-of-fold AUC of the linear probe for ctx_int with flags on
    (solid) and off (dashed), value-only baseline dotted; right, relative
    change ||h_on - h_off|| / ||h_off|| on intervened and other cells."""
    d = df[df.depth >= 0].sort_values("depth")
    on = d[d["condition"] == "flags_on"]
    off = d[d["condition"] == "flags_off"]
    ticks = on.depth.to_numpy()
    labels = on.point.to_list()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for g, ls, lab in ((on, "-", "flags on"), (off, "--", "flags off")):
        ax1.errorbar(g.depth, g.auc, yerr=g.auc_se, ls=ls, marker="o",
                     color="C0", capsize=3, label=lab)  # fmt: skip
    base = df[df.point == "value_baseline"]
    if len(base):
        ax1.axhline(float(base.auc.iloc[0]), color="grey", ls=":",
                    label="value + variable only")  # fmt: skip
    ax1.axhline(0.5, color="k", lw=0.6)
    ax1.set_ylim(0.45, 1.01)
    ax1.set_ylabel("probe AUC for ctx_int (out of fold)")
    ax1.set_title("Is the intervention flag linearly decodable?")

    ax2.plot(on.depth, on.rel_change_int, "-o", color="C1",
             label="intervened cells")  # fmt: skip
    ax2.plot(on.depth, on.rel_change_obs, "-o", color="C2",
             label="other context cells")  # fmt: skip
    ax2.set_ylabel("||h_on - h_off|| / ||h_off||")
    ax2.set_title("How much do the flags move the representation?")

    for ax in (ax1, ax2):
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("encoder depth")
        ax.legend(fontsize=8)
    _save(fig, out)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Intervention-encoding diagnostics
# --------------------------------------------------------------------------- #
# Result files of one diagnostics run with prefix P (see
# CITNP/experiments/bmrc/eval_intervention_diagnostics.sh).
PROBES_SUFFIX = "_intervention_probes.csv"
CONTEXT_EDITS_SUFFIXES = {
    "original": "_context_edits_original_normalisation.csv",
    "edited": "_context_edits_edited_normalisation.csv",
}
# Names written by CITNP.evaluation.intervention_diagnostics (not imported:
# it needs torch).
VALUE_BASELINE_POINT = "value_baseline"
EDIT_ORDER = ("relabel_observational", "move_target")
NEW_TARGET_ROLES = ("treatment", "outcome", "other")
FIGURE_FORMATS = ("png", "pdf")
SENSITIVITY_RATIO_CSV = "edit_sensitivity_ratio.csv"
# Rows of the shift figures: (name, column suffix, axis label).
SHIFT_METRICS = (
    ("mean_shift_rms", "mean_shift_rms", "RMS shift of the predictive mean"),
    ("kl", "kl_original_to_edited", "KL(original || edited), nats"),
)
_KL_LINEAR_THRESHOLD = 1e-3  # symlog axes: linear within +/- this


def _save_figure(fig, out_dir: str | Path, name: str) -> list[Path]:
    """Write fig as <out_dir>/<name>.png and .pdf, then close it."""
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    paths = [Path(out_dir) / f"{name}.{fmt}" for fmt in FIGURE_FORMATS]
    for path in paths:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote " + ", ".join(str(p) for p in paths))
    return paths


def _model_colors(labels) -> dict[str, str]:
    return {label: f"C{i}" for i, label in enumerate(labels)}


def _edits_present(edits: dict[str, pd.DataFrame]) -> list[str]:
    return [
        e for e in EDIT_ORDER if any((d.edit == e).any() for d in edits.values())
    ]


def read_intervention_diagnostics(prefixes: dict[str, str]) -> dict:
    """Result tables of labelled diagnostics runs.

    prefixes: model label -> result prefix P. Reads P_intervention_probes.csv
    and the per-row P_context_edits_{original,edited}_normalisation.csv (not
    the _summary.csv files). Returns {"probes": {label: df}, "original":
    {label: df}, "edited": {label: df}}; missing files are reported and
    left out."""
    paths = {"probes": PROBES_SUFFIX, **CONTEXT_EDITS_SUFFIXES}
    out: dict[str, dict[str, pd.DataFrame]] = {key: {} for key in paths}
    for label, prefix in prefixes.items():
        for key, suffix in paths.items():
            path = Path(prefix + suffix)
            if path.exists():
                out[key][label] = pd.read_csv(path)
            else:
                print(f"{label}: no {path}, skipping it")
    return out


def plot_probes_by_layer(
    probes: dict[str, pd.DataFrame], out_dir: str | Path
) -> list[Path]:
    """Figure probes_by_layer: one panel per probe; held-out balanced
    accuracy against layer (ordered by depth, labelled with the probe
    point), solid for true labels, dashed for shuffled labels, one colour
    per model label, dotted line at the probe's value_baseline. The y-axis
    is 0.4 - 1.0, extended down for probes whose chance level 1 / classes is
    below 0.4 (the target-variable probe), so the shuffled control stays
    visible."""
    colors = _model_colors(probes)
    names: list[str] = []
    for d in probes.values():
        names += [p for p in d.probe.unique() if p not in names]
    num_cols = min(3, len(names))
    num_rows = -(-len(names) // num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False,
                             figsize=(5.2 * num_cols, 4.2 * num_rows))  # fmt: skip
    for ax, probe in zip(axes.flat, names):
        tick_names: dict[int, set] = {}
        bottom = 0.4
        for label, df in probes.items():
            d = df[df.probe == probe]
            if d.empty:
                continue
            layers = d[d.depth >= 0]
            for variant, ls in (("true", "-"), ("shuffled", "--")):
                v = layers[layers["labels"] == variant].sort_values("depth")
                ax.plot(v.depth, v.balanced_accuracy, ls, marker="o", ms=3,
                        color=colors[label], label=f"{label}, {variant} labels")  # fmt: skip
            base = d[(d.point == VALUE_BASELINE_POINT) & (d["labels"] == "true")]
            if len(base):
                ax.axhline(float(base.balanced_accuracy.iloc[0]), ls=":",
                           color=colors[label], label=f"{label}, value baseline")  # fmt: skip
            for depth, point in zip(layers.depth, layers.point):
                tick_names.setdefault(int(depth), set()).add(str(point))
            classes = d.num_classes.max()
            if pd.notna(classes) and classes > 0:
                bottom = min(bottom, 1.0 / classes - 0.05)
        ticks = sorted(tick_names)
        ax.set_xticks(ticks)
        ax.set_xticklabels([" / ".join(sorted(tick_names[t])) for t in ticks],
                           rotation=45, ha="right", fontsize=7)  # fmt: skip
        ax.set_ylim(bottom, 1.0)
        ax.set_title(probe, fontsize=9)
        ax.set_xlabel("layer (depth)")
        ax.set_ylabel("held-out balanced accuracy")
    for ax in list(axes.flat)[len(names):]:
        ax.set_axis_off()
    axes.flat[0].legend(fontsize=6)
    return _save_figure(fig, out_dir, "probes_by_layer")


def _diagonal(ax, x, y) -> None:
    """y = x line over the range of the finite values of x and y."""
    values = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    values = values[np.isfinite(values)]
    if len(values):
        lo, hi = values.min(), values.max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)


def plot_edit_shift_scatter(
    edits: dict[str, pd.DataFrame], out_dir: str | Path
) -> list[Path]:
    """Figures edit_shift_scatter_mean_shift (signed mean shift, linear axes)
    and edit_shift_scatter_kl (KL(original || edited), log axes): one point
    per (task, query, edit, n) row; x = oracle, y = model; rows of panels =
    edit, columns = model label; colour = fraction_edited_rows; y = x line.
    Rows with KL <= 0 (Monte Carlo noise) cannot be shown on log axes; the
    panel title gives their number."""
    from matplotlib.colors import Normalize

    edit_names = _edits_present(edits)
    paths = []
    versions = (
        ("mean_shift", "mean_shift", False,
         "signed shift of the predictive mean"),
        ("kl", "kl_original_to_edited", True, "KL(original || edited), nats"),
    )  # fmt: skip
    for name, column, log, axis_label in versions:
        fig, axes = plt.subplots(
            len(edit_names), len(edits), squeeze=False, layout="constrained",
            figsize=(4.2 * len(edits) + 1, 3.8 * len(edit_names)),
        )  # fmt: skip
        for r, edit in enumerate(edit_names):
            for c, (label, df) in enumerate(edits.items()):
                ax = axes[r, c]
                d = df[df.edit == edit]
                x, y = d[f"oracle_{column}"], d[f"model_{column}"]
                title = f"{label}: {edit}"
                if log:
                    keep = (x > 0) & (y > 0)
                    if (~keep).sum():
                        title += f"\n({int((~keep).sum())} rows <= 0 not shown)"
                    d, x, y = d[keep], x[keep], y[keep]
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                ax.scatter(x, y, c=d.fraction_edited_rows, cmap="viridis",
                           vmin=0, vmax=1, s=6, alpha=0.6)  # fmt: skip
                _diagonal(ax, x, y)
                ax.set_title(title, fontsize=9)
                ax.set_xlabel(f"oracle: {axis_label}", fontsize=8)
                ax.set_ylabel(f"model: {axis_label}", fontsize=8)
        mappable = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
        fig.colorbar(
            mappable, ax=axes, label="fraction of intervened rows edited"
        )
        paths += _save_figure(fig, out_dir, f"edit_shift_scatter_{name}")
    return paths


def _quartiles(values: pd.Series) -> tuple[float, float, float]:
    q = values.dropna().quantile([0.25, 0.5, 0.75])
    return float(q[0.25]), float(q[0.5]), float(q[0.75])


def plot_edit_shift_curve(
    edits: dict[str, pd.DataFrame], out_dir: str | Path
) -> list[Path]:
    """Figure edit_shift_curve: shift against the number of edited rows n
    (log2 axis); median and interquartile band over rows, model solid,
    oracle dashed, one colour per model label; rows of panels = metric (RMS
    mean shift, KL), columns = edit. "all" is defined per row as
    num_edited_rows == num_intervened_rows and shown as its own point to the
    right, since the number of intervened rows differs between tasks."""
    colors = _model_colors(edits)
    edit_names = _edits_present(edits)
    partial_counts = sorted({
        int(n) for d in edits.values()
        for n in d.num_edited_rows[d.num_edited_rows < d.num_intervened_rows]
    })  # fmt: skip
    x_all = (np.log2(max(partial_counts)) if partial_counts else 0.0) + 1.0
    fig, axes = plt.subplots(len(SHIFT_METRICS), len(edit_names), squeeze=False,
                             figsize=(5.5 * len(edit_names), 4 * len(SHIFT_METRICS)))  # fmt: skip
    for r, (metric, column, axis_label) in enumerate(SHIFT_METRICS):
        for c, edit in enumerate(edit_names):
            ax = axes[r, c]
            for label, df in edits.items():
                d = df[df.edit == edit]
                is_all = d.num_edited_rows == d.num_intervened_rows
                styles = (("model", "-", "o"), ("oracle", "--", "s"))
                for who, ls, marker in styles:
                    col = f"{who}_{column}"
                    partial = d[~is_all].groupby("num_edited_rows")[col]
                    stats = [(np.log2(n), *_quartiles(v)) for n, v in partial]
                    if stats:
                        x, lo, mid, hi = map(np.array, zip(*stats))
                        ax.plot(x, mid, ls, marker=marker, ms=3,
                                color=colors[label], label=f"{label} {who}")  # fmt: skip
                        ax.fill_between(
                            x, lo, hi, color=colors[label], alpha=0.12
                        )
                    if is_all.any():
                        lo, mid, hi = _quartiles(d.loc[is_all, col])
                        ax.errorbar([x_all], [mid], yerr=[[mid - lo], [hi - mid]],
                                    fmt=marker, ms=5, capsize=3,
                                    color=colors[label],
                                    mfc="none" if who == "oracle" else colors[label])  # fmt: skip
            ticks = [np.log2(n) for n in partial_counts] + [x_all]
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [str(n) for n in partial_counts] + ["all"], fontsize=8
            )
            if metric == "kl":
                ax.set_yscale("symlog", linthresh=_KL_LINEAR_THRESHOLD)
            ax.set_xlabel("number of edited rows (log2; 'all' per task)")
            ax.set_ylabel(axis_label)
            ax.set_title(edit, fontsize=9)
    axes[0, 0].legend(fontsize=7)
    return _save_figure(fig, out_dir, "edit_shift_curve")


def _bootstrap_median_over_tasks(
    values: np.ndarray, tasks: np.ndarray, n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:  # fmt: skip
    """Median of values and its 95% percentile interval from n_bootstrap
    resamples of the TASKS (with replacement; every resampled task brings
    all its rows). NaN for no values."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    unique = np.unique(tasks)
    rows_of_task = [np.flatnonzero(tasks == t) for t in unique]
    medians = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        pick = rng.integers(len(unique), size=len(unique))
        rows = np.concatenate([rows_of_task[k] for k in pick])
        medians[b] = np.median(values[rows])
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(np.median(values)), float(low), float(high)


def _sensitivity_groups(df: pd.DataFrame):
    """(edit, grouping, identifiable, treatment intervened, new target role,
    row mask) of every group of `edit_sensitivity_table`; "all" where a key
    is not grouped on."""
    all_ = "all"
    for edit in EDIT_ORDER:
        in_edit = df.edit == edit
        if not in_edit.any():
            continue
        yield edit, "stratum", all_, all_, all_, in_edit
        for identifiable in (True, False):
            for treated in (True, False):
                mask = (
                    in_edit
                    & (df.identifiable_from_observational == identifiable)
                    & (df.treatment_intervened_in_context == treated)
                )
                yield edit, "stratum", identifiable, treated, all_, mask
        if edit == "move_target":
            for role in NEW_TARGET_ROLES:
                mask = (
                    in_edit
                    & (df.num_edited_rows == 1)
                    & (df.new_target_role == role)
                )
                yield edit, "new_target_role, n = 1", all_, all_, role, mask


def edit_sensitivity_table(
    edits: dict[str, pd.DataFrame],
    *,
    min_kl_standard_errors: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """The numbers behind figure edit_sensitivity_ratio.

    Per model label, edit and group (`_sensitivity_groups`: the 2 x 2 strata
    identifiable_from_observational x treatment_intervened_in_context and
    all rows; for move_target also new_target_role at num_edited_rows == 1),
    the median over rows of the per-row model / oracle ratio of
      kl:             model_kl_original_to_edited / oracle_kl_original_to_edited
      mean_shift_rms: model_mean_shift_rms / oracle_mean_shift_rms
    with a 95% interval from n_bootstrap resamples of the tasks
    (`_bootstrap_median_over_tasks`, one generator seeded with `seed`, used
    in table order). Rows are excluded where the oracle's KL is below
    min_kl_standard_errors times its Monte Carlo standard error (the oracle
    barely moves, so the ratio is noise), and, for mean_shift_rms, where the
    oracle's shift is 0. Columns: model, metric, edit, grouping,
    identifiable_from_observational, treatment_intervened_in_context,
    new_target_role, num_rows, num_excluded, num_tasks, median_ratio,
    ci_low, ci_high.
    """
    rng = np.random.default_rng(seed)
    out = []
    for label, df in edits.items():
        oracle_kl = df.oracle_kl_original_to_edited
        threshold = min_kl_standard_errors * df.oracle_kl_original_to_edited_se
        resolved = (oracle_kl > 0) & (oracle_kl >= threshold)
        ratios = {
            "kl": (df.model_kl_original_to_edited / oracle_kl, resolved),
            "mean_shift_rms": (
                df.model_mean_shift_rms / df.oracle_mean_shift_rms,
                resolved & (df.oracle_mean_shift_rms > 0),
            ),
        }
        for group in _sensitivity_groups(df):
            edit, grouping, identifiable, treated, role, mask = group
            for metric, (ratio, usable) in ratios.items():
                use = mask & usable & np.isfinite(ratio)
                median, low, high = _bootstrap_median_over_tasks(
                    ratio[use].to_numpy(), df.task[use].to_numpy(),
                    n_bootstrap, rng,
                )  # fmt: skip
                out.append({
                    "model": label, "metric": metric, "edit": edit,
                    "grouping": grouping,
                    "identifiable_from_observational": identifiable,
                    "treatment_intervened_in_context": treated,
                    "new_target_role": role,
                    "num_rows": int(mask.sum()),
                    "num_excluded": int((mask & ~use).sum()),
                    "num_tasks": int(df.task[use].nunique()),
                    "median_ratio": median, "ci_low": low, "ci_high": high,
                })  # fmt: skip
    return pd.DataFrame(out)


def _group_label(row) -> str:
    if row.grouping != "stratum":
        return f"new target = {row.new_target_role}"
    if row.identifiable_from_observational == "all":
        return "all rows"
    identifiable = row.identifiable_from_observational in (True, "True")
    treated = row.treatment_intervened_in_context in (True, "True")
    return (
        f"{'ident.' if identifiable else 'not ident.'}, "
        f"{'T int.' if treated else 'T not int.'}"
    )


def plot_edit_sensitivity_ratio(
    table: pd.DataFrame, out_dir: str | Path
) -> list[Path]:
    """Figure edit_sensitivity_ratio from `edit_sensitivity_table`: median
    per-row model / oracle ratio with its 95% task-bootstrap interval, log
    axis, line at 1; rows of panels = metric (KL, RMS mean shift), columns =
    relabel_observational strata, move_target strata, move_target by role
    of the new target (n = 1); one colour per model label. Groups whose
    median is not positive cannot be shown on the log axis."""
    colors = _model_colors(table.model.unique())
    panels = [
        (edit, grouping)
        for edit in EDIT_ORDER
        for grouping in ("stratum", "new_target_role, n = 1")
        if ((table.edit == edit) & (table.grouping == grouping)).any()
    ]
    metrics = [("kl", "KL ratio"), ("mean_shift_rms", "RMS mean-shift ratio")]
    fig, axes = plt.subplots(len(metrics), len(panels), squeeze=False,
                             figsize=(5 * len(panels), 4.2 * len(metrics)))  # fmt: skip
    num_models = len(colors)
    for r, (metric, axis_label) in enumerate(metrics):
        for c, (edit, grouping) in enumerate(panels):
            ax = axes[r, c]
            d = table[(table.metric == metric) & (table.edit == edit)
                      & (table.grouping == grouping)]  # fmt: skip
            groups = list(
                dict.fromkeys(_group_label(row) for row in d.itertuples())
            )
            for m, (label, color) in enumerate(colors.items()):
                dm = d[d.model == label]
                x = [groups.index(_group_label(row)) for row in dm.itertuples()]
                x = np.array(x) + (m - (num_models - 1) / 2) * 0.15
                mid = dm.median_ratio.to_numpy()
                ok = np.isfinite(mid) & (mid > 0)
                low = np.clip(dm.ci_low.to_numpy(), 1e-12, None)
                high = dm.ci_high.to_numpy()
                # bootstrap percentiles need not bracket the point median
                below = np.clip(np.nan_to_num(mid - low), 0, None)
                above = np.clip(np.nan_to_num(high - mid), 0, None)
                ax.errorbar(x[ok], mid[ok], yerr=[below[ok], above[ok]],
                            fmt="o", capsize=3, color=color, label=label)  # fmt: skip
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel(f"model / oracle, {axis_label}")
            ax.set_title(f"{edit} ({grouping})", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    return _save_figure(fig, out_dir, "edit_sensitivity_ratio")


_ROW_KEYS = ["task", "T", "Y", "edit", "num_edited_rows"]


def plot_normalisation_effect(
    original: dict[str, pd.DataFrame],
    edited: dict[str, pd.DataFrame],
    out_dir: str | Path,
) -> list[Path]:
    """Figure normalisation_effect: per (task, query, edit, n) row, the
    model's shift with the unedited standardisation (x, original
    normalisation run) against the shift with the edited mask's
    standardisation (y, edited normalisation run); rows of panels = signed
    mean shift (linear) and KL (symlog), columns = model label, colour =
    edit, y = x line. move_target never changes the standardisation, so
    its points lie on the line up to Monte Carlo noise."""
    labels = [label for label in original if label in edited]
    versions = (
        ("mean_shift", "signed shift of the predictive mean", False),
        ("kl_original_to_edited", "KL(original || edited), nats", True),
    )
    fig, axes = plt.subplots(len(versions), len(labels), squeeze=False,
                             figsize=(4.5 * len(labels), 4 * len(versions)))  # fmt: skip
    for c, label in enumerate(labels):
        columns = [f"model_{column}" for column, _, _ in versions]
        merged = original[label][_ROW_KEYS + columns].merge(
            edited[label][_ROW_KEYS + columns], on=_ROW_KEYS,
            suffixes=("_original", "_edited"),
        )  # fmt: skip
        for r, (column, axis_label, symlog) in enumerate(versions):
            ax = axes[r, c]
            x_all, y_all = [], []
            for k, edit in enumerate(EDIT_ORDER):
                d = merged[merged.edit == edit]
                x = d[f"model_{column}_original"]
                y = d[f"model_{column}_edited"]
                ax.scatter(x, y, s=6, alpha=0.5, color=f"C{k}", label=edit)
                x_all.append(x)
                y_all.append(y)
            _diagonal(ax, pd.concat(x_all), pd.concat(y_all))
            if symlog:
                ax.set_xscale("symlog", linthresh=_KL_LINEAR_THRESHOLD)
                ax.set_yscale("symlog", linthresh=_KL_LINEAR_THRESHOLD)
            ax.set_xlabel(f"original normalisation: {axis_label}", fontsize=8)
            ax.set_ylabel(f"edited normalisation: {axis_label}", fontsize=8)
            ax.set_title(f"{label} ({len(merged)} matched rows)", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    return _save_figure(fig, out_dir, "normalisation_effect")


def plot_secondary_discrepancy(
    edits: dict[str, pd.DataFrame], out_dir: str | Path
) -> list[Path]:
    """Figure secondary_discrepancy: boxplots of KL(oracle || model) per
    model label, on the unedited context (secondary_kl_oracle_to_model_
    original, one value per task and query) and on the edited contexts per
    edit (secondary_kl_oracle_to_model_edited); log axis, values <= 0 left
    out. Edited contexts are mislabelled, so outside the training
    distribution: secondary only."""
    fig, ax = plt.subplots(figsize=(3 + 2.5 * len(edits), 4.5))
    boxes, positions, names = [], [], []
    for m, (label, df) in enumerate(edits.items()):
        unedited = df.drop_duplicates(["task", "T", "Y"])[
            "secondary_kl_oracle_to_model_original"
        ]
        series = [("unedited", unedited)] + [
            (edit, df.loc[df.edit == edit, "secondary_kl_oracle_to_model_edited"])
            for edit in EDIT_ORDER
        ]
        for k, (name, values) in enumerate(series):
            values = values[values > 0]
            boxes.append(values.to_numpy())
            positions.append(m * (len(series) + 1) + k)
            names.append(f"{label}\n{name}")
    ax.boxplot(boxes, positions=positions, widths=0.7, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("KL(oracle || model), nats per query point")
    ax.set_title("Secondary: model-vs-oracle discrepancy (edited contexts are "
                 "out of distribution)", fontsize=9)  # fmt: skip
    return _save_figure(fig, out_dir, "secondary_discrepancy")


def plot_intervention_diagnostics(
    prefixes: dict[str, str],
    out_dir: str | Path,
    *,
    min_kl_standard_errors: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> list[Path]:
    """Every intervention-diagnostics figure (PNG and PDF) and
    edit_sensitivity_ratio.csv in out_dir, from labelled result prefixes
    (`read_intervention_diagnostics`). The context-edit figures use the
    original-normalisation run (the primary readout), except
    normalisation_effect, which compares both runs. Figures whose inputs
    are missing are skipped. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_intervention_diagnostics(prefixes)
    written: list[Path] = []
    if data["probes"]:
        written += plot_probes_by_layer(data["probes"], out_dir)
    original = data["original"]
    if original:
        written += plot_edit_shift_scatter(original, out_dir)
        written += plot_edit_shift_curve(original, out_dir)
        table = edit_sensitivity_table(
            original, min_kl_standard_errors=min_kl_standard_errors,
            n_bootstrap=n_bootstrap, seed=seed,
        )  # fmt: skip
        csv_path = out_dir / SENSITIVITY_RATIO_CSV
        table.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
        written.append(csv_path)
        overall = table[(table.grouping == "stratum")
                        & (table.identifiable_from_observational == "all")]  # fmt: skip
        for row in overall.itertuples():
            print(f"  {row.model} {row.edit} {row.metric}: excluded "
                  f"{row.num_excluded} of {row.num_rows} rows (oracle KL < "
                  f"{min_kl_standard_errors:g} standard errors)")  # fmt: skip
        written += plot_edit_sensitivity_ratio(table, out_dir)
        written += plot_secondary_discrepancy(original, out_dir)
    if any(label in data["edited"] for label in original):
        written += plot_normalisation_effect(original, data["edited"], out_dir)
    return written
