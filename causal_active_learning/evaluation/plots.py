"""Figures from the CSV tables of a run. Model-agnostic.

- <prefix>_query_kl.png: query KL to the truth against the number of
  acquisitions, one line per strategy with its 95% CI band (ABCI Fig. 4/9
  "Query KLD");
- <prefix>_calibration.png: predicted EIG against realised information gain
  of the chosen acquisitions;
- <prefix>_chosen_acquisition.png: share of each chosen acquisition per step;
- <prefix>_surrogate_task<t>_restart<r>_step<k>.png: per node, the GP
  posterior of EIG against the value, evaluated points with standard-error
  bars, the dense diagnostic and the observational EIG (horizontal line);
- `plot_panels`: ABCI Figure 4's layout, one panel per summary, the graph on
  the right.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_active_learning.evaluation.summaries import read_table, summarise

XLABEL = "number of acquisitions"
YLABEL = "query KL to the truth (nats)"
# Strategies drawn without a CI band in `plot_panels`, as ABCI does.
NO_BAND_STRATEGIES = ("observational", "random_fixed_value")
STRATEGY_COLOURS = {
    "expected_information_gain": "C0",
    "observational": "0.45",
    "random_fixed_value": "C2",
    "random": "C1",
    "treatment": "C4",
    "exhaustive_true_gain": "C3",
}


def _save(fig, path: str, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _style(strategy: str) -> dict:
    base = strategy.split("/")[0]
    colour = STRATEGY_COLOURS.get(base, "C5")
    if "/" in strategy and strategy.endswith("grid"):
        return {"color": colour, "ls": "-."}
    ls = {"observational": "--", "exhaustive_true_gain": ":"}.get(base, "-")
    return {"color": colour, "ls": ls}


def _draw_curves(ax, summary: pd.DataFrame, bands=True, all_bands=False):
    for strategy in dict.fromkeys(summary.strategy):
        d = summary[summary.strategy == strategy].sort_values("step")
        style = _style(strategy)
        ax.plot(d.step, d.mean_query_kl, label=strategy, lw=1.5, **style)
        base = strategy.split("/")[0]
        if bands and (all_bands or base not in NO_BAND_STRATEGIES):
            ax.fill_between(
                d.step,
                d.ci95_lower,
                d.ci95_upper,
                color=style["color"],
                alpha=0.15,
                lw=0,
            )


def plot_query_kl(summary: pd.DataFrame, prefix: str) -> str:
    """<prefix>_query_kl.png, one panel per setting."""
    settings = list(dict.fromkeys(summary.setting))
    fig, axes = plt.subplots(
        1,
        len(settings),
        figsize=(6.5 * len(settings), 4.5),
        squeeze=False,
        sharey=True,
    )
    for ax, setting in zip(axes[0], settings):
        d = summary[summary.setting == setting]
        _draw_curves(ax, d, all_bands=True)
        runs = int(d.num_runs.max()) if len(d) else 0
        ax.set_title(f"{setting or 'run'}: mean and 95% CI over {runs} runs")
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(YLABEL)
        ax.legend(fontsize=7)
    path = prefix + "_query_kl.png"
    _save(fig, path)
    return path


def plot_calibration(steps: pd.DataFrame, prefix: str) -> str | None:
    """<prefix>_calibration.png (None without EIG predictions)."""
    d = steps[
        pd.to_numeric(steps.predicted_information_gain, errors="coerce").notna()
    ]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for strategy, g in d.groupby("strategy", sort=False):
        ax.scatter(
            g.predicted_information_gain,
            g.realised_information_gain,
            s=10,
            alpha=0.6,
            label=strategy,
            color=_style(strategy)["color"],
        )
    top = float(
        np.nanmax(
            [
                d.predicted_information_gain.max(),
                d.realised_information_gain.max(),
                1e-6,
            ]
        )
    )
    ax.plot([0, top], [0, top], color="k", lw=0.7)
    ax.set_xlabel("predicted EIG of the chosen acquisition (nats)")
    ax.set_ylabel("realised information gain KL(after || before) (nats)")
    ax.set_title("EIG calibration")
    ax.legend(fontsize=7)
    path = prefix + "_calibration.png"
    _save(fig, path)
    return path


def _acquisition_name(label) -> str:
    text = str(label)
    if text == "observational":
        return "observational"
    return f"do(X{int(float(text)) + 1})"


def plot_chosen_acquisition_frequency(
    steps: pd.DataFrame, prefix: str
) -> str | None:
    """<prefix>_chosen_acquisition.png: share of each chosen acquisition per
    step, one panel per strategy that chooses between acquisitions."""
    d = steps[steps.step > 0]
    choosing = [
        s
        for s in dict.fromkeys(d.strategy)
        if s.split("/")[0] not in ("observational", "treatment")
    ]
    if not choosing:
        return None
    labels = sorted(
        {str(x) for x in d.chosen_node}, key=lambda x: (x == "observational", x)
    )
    fig, axes = plt.subplots(
        1, len(choosing), figsize=(4.5 * len(choosing), 4), squeeze=False
    )
    for ax, strategy in zip(axes[0], choosing):
        g = d[d.strategy == strategy]
        share = pd.crosstab(g.step, g.chosen_node.astype(str), normalize="index")
        bottom = np.zeros(len(share))
        for label in labels:
            if label not in share:
                continue
            ax.bar(
                share.index,
                share[label],
                bottom=bottom,
                label=_acquisition_name(label),
            )
            bottom += share[label].to_numpy()
        ax.set_xlabel("step")
        ax.set_ylabel("share of chosen acquisitions")
        ax.set_title(strategy, fontsize=9)
        ax.legend(fontsize=7)
    path = prefix + "_chosen_acquisition.png"
    _save(fig, path)
    return path


def plot_value_surrogate(
    trace: pd.DataFrame,
    surrogate: pd.DataFrame,
    scores: pd.DataFrame,
    prefix: str,
    *,
    task,
    restart,
    step: int,
    strategy: str,
    dense: pd.DataFrame | None = None,
) -> str | None:
    """Per node at one (task, restart, step, strategy): the final GP posterior
    of EIG against the value, the evaluated points (two standard errors), the
    re-estimated recommendation (star), the dense diagnostic (both noise
    modes) and the observational candidate's EIG (horizontal line)."""

    def at(df):
        if df is None or df.empty:
            return df
        return df[
            (df.task == task)
            & (df.restart == restart)
            & (df.step == step)
            & (df.strategy == strategy)
        ]

    t, s, sc, dn = at(trace), at(surrogate), at(scores), at(dense)
    if t is None or t.empty:
        return None
    nodes = sorted(t.node.unique())
    fig, axes = plt.subplots(
        1, len(nodes), figsize=(4.5 * len(nodes), 4), squeeze=False
    )
    observational = None
    if sc is not None and not sc.empty:
        o = sc[sc.node.astype(str) == "observational"]
        if not o.empty:
            observational = float(o.expected_information_gain.iloc[0])
    for ax, node in zip(axes[0], nodes):
        if s is not None and not s.empty:
            sn = s[s.node == node]
            ax.plot(sn.value, sn.posterior_mean, color="C0", label="GP mean")
            ax.fill_between(
                sn.value,
                sn.posterior_mean - 2 * sn.posterior_standard_deviation,
                sn.posterior_mean + 2 * sn.posterior_standard_deviation,
                color="C0",
                alpha=0.2,
                label="GP +- 2 sd",
            )
        tn = t[t.node == node]
        ax.errorbar(
            tn.value,
            tn.expected_information_gain,
            yerr=2 * tn.standard_error,
            fmt="o",
            ms=4,
            color="C3",
            label="evaluations",
        )
        if dn is not None and not dn.empty:
            for mode, ls in (("independent", "--"), ("shared", ":")):
                dd = dn[(dn.node == node) & (dn.noise_mode == mode)]
                if not dd.empty:
                    ax.plot(
                        dd.value,
                        dd.expected_information_gain,
                        color="0.4",
                        lw=1,
                        ls=ls,
                        label=f"dense ({mode} noise)",
                    )
        if sc is not None and not sc.empty:
            r = sc[sc.node.astype(str) == str(node)]
            ax.plot(
                r.value,
                r.expected_information_gain,
                "*",
                ms=12,
                color="C2",
                label="re-estimated",
            )
        if observational is not None:
            ax.axhline(observational, color="0.3", lw=1, label="observational")
        ax.set_xlabel(f"do-value of X{int(node) + 1}")
        ax.set_ylabel("EIG (nats)")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"task {task}, restart {restart}, step {step}, {strategy}", fontsize=9
    )
    path = f"{prefix}_surrogate_task{task}_restart{restart}_step{step}.png"
    _save(fig, path)
    return path


def plot_active_learning_all(prefix: str) -> list[str]:
    """Every per-run figure from the CSVs written with `prefix`; the
    surrogate figure for the first task and restart at every step present in
    the surrogate table. Returns the files written."""
    steps = read_table(prefix + "_steps.csv")
    if steps.empty:
        return []
    summary = read_table(prefix + "_summary.csv")
    if summary.empty:
        summary = summarise(steps)
    written = [plot_query_kl(summary, prefix)]
    for path in (
        plot_calibration(steps, prefix),
        plot_chosen_acquisition_frequency(steps, prefix),
    ):
        if path:
            written.append(path)
    trace = read_table(prefix + "_bayesian_optimisation_trace.csv")
    surrogate = read_table(prefix + "_surrogate.csv")
    scores = read_table(prefix + "_scores.csv")
    dense = read_table(prefix + "_dense_value_diagnostic.csv")
    if trace.empty or surrogate.empty:
        return written
    first = surrogate.iloc[0]
    strategy, task, restart = first.strategy, first.task, first.restart
    wanted = surrogate[
        (surrogate.task == task)
        & (surrogate.restart == restart)
        & (surrogate.strategy == strategy)
    ].step.unique()
    for step in sorted(wanted):
        path = plot_value_surrogate(
            trace,
            surrogate,
            scores,
            prefix,
            task=task,
            restart=restart,
            step=int(step),
            strategy=strategy,
            dense=dense,
        )
        if path:
            written.append(path)
    return written


# --------------------------------------------------------------------------- #
# ABCI Figure 4 layout
# --------------------------------------------------------------------------- #
def _layered_positions(
    num_variables: int, edges
) -> dict[int, tuple[float, float]]:
    """Node positions by longest-path depth from the roots (top to bottom)."""
    depth = dict.fromkeys(range(num_variables), 0)
    for _ in range(num_variables):
        for parent, child in edges:
            depth[child] = max(depth[child], depth[parent] + 1)
    layers: dict[int, list[int]] = {}
    for node, d in depth.items():
        layers.setdefault(d, []).append(node)
    positions: dict[int, tuple[float, float]] = {}
    for d, nodes in layers.items():
        for i, node in enumerate(sorted(nodes)):
            positions[node] = (i - (len(nodes) - 1) / 2, float(-d))
    return positions


def draw_graph(ax, num_variables, edges, treatment, outcome, intervenable_nodes):
    """The DAG with the treatment and outcome coloured and non-intervenable
    nodes drawn with a dashed outline."""
    pos = _layered_positions(num_variables, edges)
    for parent, child in edges:
        (x0, y0), (x1, y1) = pos[parent], pos[child]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color="0.3", lw=1, shrinkA=13, shrinkB=13
            ),
        )
    intervenable = (
        set(range(num_variables))
        if intervenable_nodes is None
        else set(intervenable_nodes)
    )
    for node, (x, y) in pos.items():
        face = {treatment: "#9ecae1", outcome: "#fdae6b"}.get(node, "white")
        circle = plt.Circle(
            (x, y),
            0.28,
            facecolor=face,
            edgecolor="k",
            lw=1.2,
            ls="-" if node in intervenable else "--",
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x, y, f"X{node + 1}", ha="center", va="center", zorder=4, fontsize=9
        )
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_panels(
    summaries: Sequence[pd.DataFrame],
    experiments: Sequence[dict],
    titles: Sequence[str],
    out: str,
    *,
    all_bands: bool = False,
) -> str:
    """ABCI Figure 4's layout: one panel per summary, query KL against the
    number of acquisitions, mean line and 95% CI band per strategy (bands
    omitted for NO_BAND_STRATEGIES unless `all_bands`), shared y axis, legend
    below, and the graph of each experiment on the right (treatment blue,
    outcome orange, non-intervenable nodes dashed). `experiments` are dicts
    with num_variables, edges, treatment, outcome, intervenable_nodes. A
    caption notes when the EIG strategy had no observational candidate."""
    n = len(summaries)
    fig = plt.figure(figsize=(5.5 * n + 2.5, 4.8))
    grid = fig.add_gridspec(1, n + 1, width_ratios=[4] * n + [1.6])
    first = None
    for i, (summary, title) in enumerate(zip(summaries, titles)):
        ax = fig.add_subplot(grid[0, i], sharey=first)
        first = first or ax
        _draw_curves(ax, summary, all_bands=all_bands)
        ax.set_title(title)
        ax.set_xlabel(XLABEL)
        if i == 0:
            ax.set_ylabel(YLABEL)
    graph_grid = grid[0, n].subgridspec(len(experiments), 1)
    for j, spec in enumerate(experiments):
        gax = fig.add_subplot(graph_grid[j, 0])
        draw_graph(
            gax,
            spec["num_variables"],
            spec["edges"],
            spec["treatment"],
            spec["outcome"],
            spec.get("intervenable_nodes"),
        )
        gax.set_title(titles[j] if j < len(titles) else "", fontsize=8)
    handles, labels = first.get_legend_handles_labels() if first else ([], [])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        fontsize=8,
        bbox_to_anchor=(0.45, -0.02),
    )
    caption = []
    for summary in summaries:
        eig = summary[
            summary.strategy.str.startswith("expected_information_gain")
        ]
        if not eig.empty and (
            eig.fraction_observational_candidate_available.max() < 1.0
        ):
            caption.append(
                "The EIG strategy could not score the observational candidate "
                "(the model cannot predict observational rows); the random "
                "baselines and OBS could choose it."
            )
            break
    if caption:
        fig.text(0.01, -0.08, caption[0], fontsize=8, ha="left")
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
