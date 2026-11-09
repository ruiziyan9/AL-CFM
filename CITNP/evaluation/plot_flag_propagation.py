"""Figures of the flag-propagation diagnostic. Reads the files written by
`flag_propagation.save_flag_propagation` only; never calls a model.

One run directory (`plot_flag_propagation(run_dir)`), written next to the data:
    flag_propagation_A_example<i>.png   per-stage [N, D'] sensitivity heatmaps
                                        of one source (absolute and
                                        stage-normalised), query strip below
    flag_propagation_B_curves.png       category mean vs stage, 95% context
                                        bootstrap intervals, noise floor in grey
    flag_propagation_D_injection.png    layer-wise only: all / first-only /
                                        later-only injections per category
    flag_propagation_E_predictive_kl.png predictive KL by ablation mode and by
                                        source regime size
    flag_propagation_F_net_change.png   net sensitivity change per stage
Several run directories (`plot_architecture_comparison(run_dirs, out_dir)`),
which must share the same source manifest:
    flag_propagation_C_architectures.png  stage-normalised category curves per
                                        architecture on identical axes, and the
                                        initial source magnitude
    flag_propagation_E_predictive_kl.png  as above, one column per run

Curves aggregate over sources within each context first, then across
contexts; intervals bootstrap contexts (never cells or sources). Only
`all_injections` rows enter B, C, E and F.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from CITNP.evaluation.flag_propagation import (
    METADATA_CATEGORIES,
    METADATA_FILE,
    ORDINARY_CATEGORIES,
    PREDICTIONS_FILE,
    QUERY_CATEGORIES,
    SOURCES_FILE,
    SUMMARY_FILE,
    EXAMPLES_FILE,
    INJECTION_MODES,
)  # fmt: skip
from CITNP.evaluation.plots import _save

# Validated categorical palette, assigned in a fixed order (never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fmt: skip
CATEGORY_STYLE = {
    **{c: (PALETTE[i], "-") for i, c in enumerate(ORDINARY_CATEGORIES)},
    "metadata_source_rows": (PALETTE[6], "-"),
    "metadata_other_rows": (PALETTE[7], "-"),
    "query_cells": (PALETTE[6], "--"),
    "query_metadata": (PALETTE[7], "--"),
}
FLOOR_COLOR = "#8a8984"
HEATMAP_CMAP = "Blues"  # one hue, light -> dark
STAGE_METRIC = "mean_stage_normalised_l2"
STAGE_LABEL = "||Δh|| / stage median ||h||"
MAX_HEATMAP_ROWS = 48
C_COLUMNS = 6
N_BOOT = 1000


# --------------------------------------------------------------------------- #
# Loading and aggregation
# --------------------------------------------------------------------------- #
def load_run(run_dir) -> dict:
    run = Path(run_dir)
    out = {
        "dir": run,
        "summary": pd.read_csv(run / SUMMARY_FILE),
        "predictions": pd.read_csv(run / PREDICTIONS_FILE),
        "sources": pd.read_csv(run / SOURCES_FILE),
        "metadata": json.loads((run / METADATA_FILE).read_text()),
    }
    out["label"] = run_label(out["metadata"])
    return out


def run_label(meta: dict) -> str:
    if meta["architecture"] == "layerwise":
        shared = "shared" if meta.get("share_intervention_encoder") else "separate"
        return f"layerwise ({shared} flag MLP)"
    return meta["architecture"].replace("_", "-")


def load_examples(path) -> list[dict]:
    """Examples saved in flag_propagation_examples.npz."""
    with np.load(path, allow_pickle=False) as z:
        n = int(z["num_examples"])
        stage_names = [str(s) for s in z["stage_names"]]
        out = []
        for i in range(n):
            prefix = f"example{i}_"
            ex = {k[len(prefix) :]: z[k] for k in z.files if k.startswith(prefix)}
            for k in ("context_id", "source_id", "ablation_mode", "has_metadata"):
                ex[k] = ex[k].item()
            ex["stage_names"] = stage_names
            out.append(ex)
    return out


def context_bootstrap(
    df: pd.DataFrame, value: str, num_stages: int, *, n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """(mean, lower, upper) [num_stages] over contexts with a 95% percentile
    bootstrap over contexts. Rows are averaged over sources within each
    context first; contexts where the value is missing are skipped."""
    per_context = df.groupby(["context_id", "stage_index"])[value].mean().unstack()
    mat = per_context.reindex(columns=range(num_stages)).to_numpy(float)
    nan = np.full(num_stages, np.nan)
    if mat.size == 0:
        return nan, nan, nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        mean = np.nanmean(mat, axis=0)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(mat), size=(n_boot, len(mat)))
        boot = np.nanmean(mat[idx], axis=1)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return mean, lo, hi


def _stage_axis(ax, stage_names: list[str]) -> None:
    """Stage ticks, node stages shaded, block boundaries as dotted lines."""
    x = np.arange(len(stage_names))
    for i, s in enumerate(stage_names):
        if s.endswith("/node"):
            ax.axvspan(i - 0.5, i + 0.5, color="#f0efec", zorder=0, lw=0)
        if s.endswith("/sample"):
            ax.axvline(i - 0.5, color="#c3c2b7", lw=0.8, ls=":", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.5, len(stage_names) - 0.5)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _floor(ax, df, floor_col, categories, num_stages, label=True) -> None:
    """Grey band from 0 to the largest category noise floor at each stage."""
    curves = [
        context_bootstrap(df[df.category == c], floor_col, num_stages, n_boot=1)[0]
        for c in categories
        if (df.category == c).any()
    ]
    if not curves:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        top = np.nanmax(np.vstack(curves), axis=0)
    x = np.arange(num_stages)
    ax.fill_between(x, 0, np.nan_to_num(top), color=FLOOR_COLOR, alpha=0.25,
                    lw=0, label="noise floor (identical-mask control)" if label else None)  # fmt: skip
    ax.plot(x, top, color=FLOOR_COLOR, ls="--", lw=1)


def _category_curves(ax, df, categories, value, num_stages, *, floor_col=None):
    x = np.arange(num_stages)
    for c in categories:
        d = df[df.category == c]
        if d.empty or d[value].isna().all():
            continue
        mean, lo, hi = context_bootstrap(d, value, num_stages)
        color, ls = CATEGORY_STYLE[c]
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(x, mean, color=color, ls=ls, lw=2, marker="o", ms=4,
                label=c.replace("_", " "))  # fmt: skip
    if floor_col is not None:
        _floor(ax, df, floor_col, categories, num_stages)


def _panels(meta: dict) -> list[tuple[str, tuple[str, ...]]]:
    panels = [("context cells", ORDINARY_CATEGORIES)]
    if meta.get("metadata_token"):
        panels.append(("metadata tokens", METADATA_CATEGORIES))
        panels.append(("query tokens", QUERY_CATEGORIES))
    else:
        panels.append(("query tokens", ("query_cells",)))
    return panels


def _all_injections(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.injection_mode == "all_injections"]


def _ablation_modes(df: pd.DataFrame) -> list[str]:
    return [m for m in ("single_flag", "regime") if (df.ablation_mode == m).any()]


# --------------------------------------------------------------------------- #
# Figure A: example heatmaps
# --------------------------------------------------------------------------- #
def _display_rows(labels: np.ndarray, source_rows: set[int]) -> np.ndarray:
    """Context rows sorted by regime (display only); if there are more than
    MAX_HEATMAP_ROWS, keep the source rows and an even subset of each
    regime."""
    order = np.argsort(labels, kind="stable")
    if len(order) <= MAX_HEATMAP_ROWS:
        return order
    keep = set(source_rows)
    budget = MAX_HEATMAP_ROWS - len(keep)
    for g in np.unique(labels):
        rows = [r for r in order if labels[r] == g and r not in keep]
        share = max(1, round(budget * len(rows) / len(order)))
        keep.update(rows[:: max(1, len(rows) // share)][:share])
    return np.array([r for r in order if r in keep])


def plot_example(ex: dict, path) -> None:
    """Per-stage sensitivity of one source: absolute (top) and
    stage-normalised (bottom) context heatmaps [rows, D'] with the query
    strip (rows = variables, plus META for extra-node; columns = query
    do-values) under each. One colour scale per version."""
    stages = ex["stage_names"]
    S = len(stages)
    labels = ex["regime_labels"]
    src = {(int(r), int(k)) for r, k in ex["source_cells"]}
    rows = _display_rows(labels, {r for r, _ in src})
    has_meta = bool(ex["has_metadata"])
    D = ex["original_mask"].shape[1]
    cols = [f"x{j}" for j in range(D)] + (["META"] if has_meta else [])
    n_q = ex["query_abs_l2"].shape[1]
    fig = plt.figure(figsize=(1.6 * S + 1.5, 0.12 * len(rows) * 2 + 4.5))
    grid = fig.add_gridspec(
        4, S + 1, width_ratios=[1] * S + [0.08],
        height_ratios=[len(rows), len(cols) + 1, len(rows), len(cols) + 1],
        hspace=0.35, wspace=0.15,
    )  # fmt: skip
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad("#e6e5e0")
    for v, (kind, title) in enumerate(
        (("abs_l2", "||Δh||"), ("stage_normalised_l2", STAGE_LABEL))
    ):
        ctx = ex[f"context_{kind}"][:, rows]  # [S, rows, D']
        qry = np.transpose(ex[f"query_{kind}"], (0, 2, 1))  # [S, D', n_q]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vmax = float(np.nanmax([np.nanmax(ctx), np.nanmax(qry), 1e-12]))
        for i, s in enumerate(stages):
            ax = fig.add_subplot(grid[2 * v, i])
            im = ax.imshow(ctx[i], aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                           interpolation="nearest")  # fmt: skip
            for y, r in enumerate(rows):
                for k in range(D):
                    if (int(r), k) in src:
                        ax.add_patch(Rectangle((k - 0.5, y - 0.5), 1, 1, fill=False,
                                               ec="#e34948", lw=1.2))  # fmt: skip
                if has_meta and any(int(r) == sr for sr, _ in src):
                    ax.add_patch(Rectangle((D - 0.5, y - 0.5), 1, 1, fill=False,
                                           ec="#e34948", lw=1.2, ls=":"))  # fmt: skip
            sorted_labels = labels[rows]
            for y in np.flatnonzero(np.diff(sorted_labels)) + 0.5:
                ax.axhline(y, color="#0b0b0b", lw=0.6)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, fontsize=6)
            ax.set_yticks([])
            ax.set_title(s, fontsize=8)
            if i == 0:
                ax.set_ylabel(f"{title}\ncontext rows (by regime)", fontsize=7)
            qa = fig.add_subplot(grid[2 * v + 1, i])
            qa.imshow(qry[i], aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                      interpolation="nearest")  # fmt: skip
            qa.set_yticks(range(len(cols)))
            qa.set_yticklabels(cols if i == 0 else [], fontsize=6)
            qa.set_xticks(range(n_q))
            qa.set_xticklabels([])
            if i == 0:
                qa.set_ylabel("queries", fontsize=7)
                qa.set_xlabel("do-values", fontsize=6)
        cb = fig.add_subplot(grid[2 * v, S])
        fig.colorbar(im, cax=cb)
        cb.tick_params(labelsize=6)
    fig.suptitle(
        f"context {ex['context_id']}, source {ex['source_id']} "
        f"({ex['ablation_mode'].replace('_', ' ')}; removed flags outlined; "
        "lines = regime boundaries; grey = undefined)",
        fontsize=9,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figures B, D, F
# --------------------------------------------------------------------------- #
def plot_category_curves(run: dict, path, *, value=STAGE_METRIC, net_change=False):
    """B (or F with net_change): category means vs stage, rows = ablation
    modes, columns = context / metadata / query panels."""
    meta, df = run["metadata"], _all_injections(run["summary"])
    stages = meta["stage_names"]
    if net_change:
        value = f"net_change_{STAGE_METRIC}"
    modes, panels = _ablation_modes(df), _panels(meta)
    fig, axes = plt.subplots(len(modes), len(panels), squeeze=False, sharex=True,
                             figsize=(5 * len(panels), 3.6 * len(modes)))  # fmt: skip
    for r, mode in enumerate(modes):
        d = df[df.ablation_mode == mode]
        for c, (title, cats) in enumerate(panels):
            ax = axes[r, c]
            _category_curves(ax, d, cats, value, len(stages),
                             floor_col=None if net_change else "noise_floor_mean_stage_normalised_l2")  # fmt: skip
            if net_change:
                ax.axhline(0, color="#52514e", lw=0.8)
            _stage_axis(ax, stages)
            ax.set_title(f"{mode.replace('_', ' ')} ablation: {title}", fontsize=9)
            if c == 0:
                ax.set_ylabel("net sensitivity change" if net_change else STAGE_LABEL, fontsize=8)  # fmt: skip
            ax.legend(fontsize=6, frameon=False)
    fig.suptitle(
        f"{run['label']}: "
        + ("net sensitivity change between stages (not attention pathways)"
           if net_change else "sensitivity to removed flags (mean, 95% context bootstrap)")
        + "; shaded columns = node-attention stages",
        fontsize=10,
    )  # fmt: skip
    _save(fig, path)


def plot_injection_decomposition(run: dict, path) -> None:
    """D: per category (columns) and ablation mode (rows), the three
    injection modes of a layer-wise model."""
    meta, df = run["metadata"], run["summary"]
    stages = meta["stage_names"]
    cats = (*ORDINARY_CATEGORIES, "query_cells")
    modes = _ablation_modes(df)
    fig, axes = plt.subplots(len(modes), len(cats), squeeze=False, sharex=True,
                             sharey="row", figsize=(2.9 * len(cats), 3.2 * len(modes)))  # fmt: skip
    x = np.arange(len(stages))
    for r, mode in enumerate(modes):
        for c, cat in enumerate(cats):
            ax = axes[r, c]
            d = df[(df.ablation_mode == mode) & (df.category == cat)]
            if d[STAGE_METRIC].isna().all():
                ax.text(0.5, 0.5, "no destinations", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="#52514e")  # fmt: skip
            for i, inj in enumerate(INJECTION_MODES):
                di = d[d.injection_mode == inj]
                if di.empty or di[STAGE_METRIC].isna().all():
                    continue
                mean, lo, hi = context_bootstrap(di, STAGE_METRIC, len(stages))
                ax.fill_between(x, lo, hi, color=PALETTE[i], alpha=0.15, lw=0)
                ax.plot(x, mean, color=PALETTE[i], lw=2, marker="o", ms=3,
                        label=inj.replace("_", " "))  # fmt: skip
            _floor(ax, d[d.injection_mode == "all_injections"],
                   "noise_floor_mean_stage_normalised_l2", (cat,), len(stages))  # fmt: skip
            _stage_axis(ax, stages)
            ax.set_title(f"{mode.replace('_', ' ')}: {cat.replace('_', ' ')}", fontsize=7)
            if c == 0:
                ax.set_ylabel(STAGE_LABEL, fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(
        f"{run['label']}: injection decomposition (first = embedding + "
        "inject_sample[0]; 'embed' is recorded before inject_sample[0])",
        fontsize=10,
    )
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Figure E: output-level change
# --------------------------------------------------------------------------- #
def _per_source_kl(predictions: pd.DataFrame) -> pd.DataFrame:
    d = _all_injections(predictions)
    keys = ["context_id", "source_id", "ablation_mode", "source_regime_size"]
    return d.groupby(keys, as_index=False)[
        ["predictive_kl", "noise_floor_predictive_kl"]
    ].mean()


def plot_predictive_kl(runs: list[dict], path) -> None:
    """E: per-source predictive KL (mean over query points) by ablation mode
    (top) and by source regime size (bottom), one column per run."""
    fig, axes = plt.subplots(2, len(runs), squeeze=False,
                             figsize=(5 * len(runs), 7.5), sharey="row")  # fmt: skip
    for c, run in enumerate(runs):
        kl = _per_source_kl(run["predictions"])
        floor = float(kl.noise_floor_predictive_kl.abs().max()) if len(kl) else 0.0
        for r, by in enumerate(("ablation_mode", "source_regime_size")):
            ax = axes[r, c]
            order = {"single_flag": 0, "regime": 1}
            groups = sorted(kl[by].unique(), key=lambda g: order.get(g, g))
            data = [kl.loc[kl[by] == g, "predictive_kl"].to_numpy() for g in groups]
            if data:
                ax.boxplot(data, widths=0.5, medianprops=dict(color=PALETTE[0], lw=2),
                           flierprops=dict(ms=3))  # fmt: skip
                ax.set_xticks(range(1, len(groups) + 1))
                ax.set_xticklabels([str(g).replace("_", " ") for g in groups], fontsize=8)
            ax.axhline(floor, color=FLOOR_COLOR, ls="--", lw=1,
                       label="noise floor (largest |control KL|)")  # fmt: skip
            ax.set_yscale("symlog", linthresh=1e-4)
            values = np.concatenate([*data, [0.0, floor]]) if data else np.zeros(1)
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
            ax.set_ylim(min(low * 3, 0.0), max(high * 3, 1e-3))
            ax.set_xlabel(by.replace("_", " "), fontsize=8)
            ax.set_ylabel("KL(p(·|M) || p(·|M⁻)) [nats]", fontsize=8)
            ax.set_title(run["label"], fontsize=9)
            ax.legend(fontsize=6, frameon=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.suptitle("Output-level change from removing flags (all injections)", fontsize=10)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Figure C: architecture comparison
# --------------------------------------------------------------------------- #
MANIFEST_KEYS = ["context_id", "context_fingerprint", "source_id", "ablation_mode",
                 "source_variable", "source_rows", "source_regime"]  # fmt: skip


def check_same_manifest(runs: list[dict]) -> None:
    ref = runs[0]["sources"][MANIFEST_KEYS].astype(str).reset_index(drop=True)
    for run in runs[1:]:
        other = run["sources"][MANIFEST_KEYS].astype(str).reset_index(drop=True)
        if not ref.equals(other):
            raise ValueError(
                f"{run['dir']} used a different source manifest than "
                f"{runs[0]['dir']}; rerun it with --source_manifest"
            )
        if run["metadata"]["stage_names"] != runs[0]["metadata"]["stage_names"]:
            raise ValueError("runs have different encoder stages; cannot align them")


def plot_architecture_comparison_curves(runs: list[dict], path) -> None:
    """C: stage-normalised category curves, one line per run, identical
    y-axes within each ablation mode; the last panel of each mode is the
    initial source magnitude per run. Panels wrap at C_COLUMNS per row."""
    check_same_manifest(runs)
    stages = runs[0]["metadata"]["stage_names"]
    cats = [*ORDINARY_CATEGORIES, "query_cells"]
    if any(r["metadata"].get("metadata_token") for r in runs):
        cats += [*METADATA_CATEGORIES, "query_metadata"]
    modes = _ablation_modes(runs[0]["summary"])
    rows_per_mode = -(-(len(cats) + 1) // C_COLUMNS)
    fig, axes = plt.subplots(len(modes) * rows_per_mode, C_COLUMNS, squeeze=False,
                             figsize=(3.0 * C_COLUMNS, 3.2 * len(modes) * rows_per_mode))  # fmt: skip
    x = np.arange(len(stages))
    for r, mode in enumerate(modes):
        mode_axes = axes[r * rows_per_mode : (r + 1) * rows_per_mode].ravel()
        for c, cat in enumerate(cats):
            ax = mode_axes[c]
            for i, run in enumerate(runs):
                d = _all_injections(run["summary"])
                d = d[(d.ablation_mode == mode) & (d.category == cat)]
                if d.empty or d[STAGE_METRIC].isna().all():
                    continue
                mean, lo, hi = context_bootstrap(d, STAGE_METRIC, len(stages))
                ax.fill_between(x, lo, hi, color=PALETTE[i], alpha=0.15, lw=0)
                ax.plot(x, mean, color=PALETTE[i], lw=2, marker="o", ms=3, label=run["label"])  # fmt: skip
                _floor(ax, d, "noise_floor_mean_stage_normalised_l2", (cat,),
                       len(stages), label=i == 0)  # fmt: skip
            _stage_axis(ax, stages)
            ax.set_title(f"{mode.replace('_', ' ')}: {cat.replace('_', ' ')}", fontsize=7)
            if c % C_COLUMNS == 0:
                ax.set_ylabel(STAGE_LABEL, fontsize=7)
        curve_axes = mode_axes[: len(cats)]
        lims = [a.get_ylim() for a in curve_axes]
        for a in curve_axes:  # identical axes within the ablation mode
            a.set_ylim(min(lo for lo, _ in lims), max(hi for _, hi in lims))
        ax = mode_axes[len(cats)]
        data = []
        for run in runs:
            d = _all_injections(run["summary"])
            d = d[(d.ablation_mode == mode) & (d.stage_index == 0)]
            data.append(d.groupby("context_id").initial_source_l2.mean().to_numpy())
        ax.boxplot(data, widths=0.5, medianprops=dict(color=PALETTE[0], lw=2))
        ax.set_xticks(range(1, len(runs) + 1))
        ax.set_xticklabels([run["label"] for run in runs], rotation=30, ha="right", fontsize=6)
        ax.set_title(f"{mode.replace('_', ' ')}: initial source magnitude at embed", fontsize=7)
        ax.set_ylabel("||Δh|| (absolute)", fontsize=7)
        for a in mode_axes[len(cats) + 1 :]:
            a.set_visible(False)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Architecture comparison on the same contexts and sources "
                 "(mean, 95% context bootstrap; grey = noise floor; shaded "
                 "columns = node-attention stages)", fontsize=10)  # fmt: skip
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def plot_flag_propagation(run_dir) -> list[Path]:
    """All single-run figures (A, B, D if layer-wise, E, F) into `run_dir`."""
    run = load_run(run_dir)
    out = Path(run_dir)
    written = []
    examples = out / EXAMPLES_FILE
    if examples.exists():
        for i, ex in enumerate(load_examples(examples)):
            written.append(out / f"flag_propagation_A_example{i}.png")
            plot_example(ex, written[-1])
    written.append(out / "flag_propagation_B_curves.png")
    plot_category_curves(run, written[-1])
    if (run["summary"].injection_mode != "all_injections").any():
        written.append(out / "flag_propagation_D_injection.png")
        plot_injection_decomposition(run, written[-1])
    written.append(out / "flag_propagation_E_predictive_kl.png")
    plot_predictive_kl([run], written[-1])
    written.append(out / "flag_propagation_F_net_change.png")
    plot_category_curves(run, written[-1], net_change=True)
    return written


def plot_architecture_comparison(run_dirs, out_dir) -> list[Path]:
    """Figures C and E across runs that share one source manifest."""
    runs = [load_run(d) for d in run_dirs]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "flag_propagation_C_architectures.png",
             out / "flag_propagation_E_predictive_kl.png"]  # fmt: skip
    plot_architecture_comparison_curves(runs, paths[0])
    plot_predictive_kl(runs, paths[1])
    return paths
