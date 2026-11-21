"""Figures of the active learning evaluation, from its CSV tables, in the
style of the k-shot figures (CITNP/evaluation/plots.py).

- <prefix>_kl_to_truth.png: KL(truth || model) against the number of
  experiments, one line per strategy, mean and 95% band over tasks and seeds
  (as ABCI Fig. 4 / Fig. 9, "Query KLD").
- <prefix>_calibration.png: predicted EIG against realised information gain
  of the chosen experiments (EIG strategies).
- <prefix>_chosen_node.png: how often each node is chosen at each step.
- <prefix>_surrogate_task<i>_seed<s>_step<k>.png: per node, the final GP
  posterior mean and band of EIG against the value, with the evaluated
  points, the recommended value, and the dense diagnostic if present.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CITNP.evaluation.plots import _save

KL_XLAB = "number of experiments"
_BAND_Z = 1.96  # 95% normal band of the mean


def _strategy_style(label: str, order: list[str]) -> dict:
    if label == "observational":
        return {"color": "0.45", "ls": "--"}
    if label.startswith("exhaustive_true_gain"):
        return {"color": f"C{order.index(label) % 10}", "ls": ":"}
    return {"color": f"C{order.index(label) % 10}", "ls": "-"}


def kl_curves(steps: pd.DataFrame) -> pd.DataFrame:
    """KL to the truth per (task, seed, strategy) against the number of
    experiments 0..budget (0 = the initial context)."""
    start = steps[steps.step == 1].assign(
        experiments=0, kl=lambda d: d.kl_to_truth_before
    )
    after = steps.assign(experiments=steps.step, kl=steps.kl_to_truth_after)
    cols = ["task", "seed", "strategy", "experiments", "kl"]
    return pd.concat([start[cols], after[cols]], ignore_index=True)


def plot_kl_to_truth(steps: pd.DataFrame, prefix: str) -> None:
    """<prefix>_kl_to_truth.png."""
    curves = kl_curves(steps)
    order = list(dict.fromkeys(steps.strategy))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label in order:
        g = curves[curves.strategy == label].groupby("experiments")["kl"]
        mean, sem = g.mean(), g.sem().fillna(0.0)
        style = _strategy_style(label, order)
        ax.plot(mean.index, mean.values, marker="o", ms=3, label=label, **style)
        ax.fill_between(
            mean.index, mean - _BAND_Z * sem, mean + _BAND_Z * sem,
            color=style["color"], alpha=0.15,
        )  # fmt: skip
    units = curves.groupby(["task", "seed"]).ngroups
    ax.set_xticks(sorted(curves.experiments.unique()))
    ax.set_xlabel(KL_XLAB)
    ax.set_ylabel("KL(truth || model), nats per query point")
    ax.set_title(
        f"query KL to the truth, mean and 95% band over {units} task x seed"
    )
    ax.legend(fontsize=8)
    _save(fig, prefix + "_kl_to_truth.png")


def plot_calibration(steps: pd.DataFrame, prefix: str) -> bool:
    """<prefix>_calibration.png (skipped without EIG strategies); True if
    written."""
    d = steps[steps.predicted_information_gain.notna()]
    if d.empty:
        return False
    order = list(dict.fromkeys(steps.strategy))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, g in d.groupby("strategy", sort=False):
        ax.scatter(
            g.predicted_information_gain, g.realised_information_gain, s=10,
            alpha=0.6, label=label, color=_strategy_style(label, order)["color"],
        )  # fmt: skip
    top = float(
        np.nanmax([d.predicted_information_gain.max(),
                   d.realised_information_gain.max(), 1e-6])
    )  # fmt: skip
    ax.plot([0, top], [0, top], color="k", lw=0.7)
    ax.set_xlabel("predicted EIG of the chosen experiment (nats)")
    ax.set_ylabel("realised information gain KL(after || before) (nats)")
    ax.set_title("EIG calibration")
    ax.legend(fontsize=8)
    _save(fig, prefix + "_calibration.png")
    return True


def plot_chosen_node_frequency(steps: pd.DataFrame, prefix: str) -> bool:
    """<prefix>_chosen_node.png: share of each chosen node per step, one
    panel per strategy that chooses between nodes; True if written."""
    choosing = [
        s for s in dict.fromkeys(steps.strategy)
        if s not in ("observational", "treatment")
    ]  # fmt: skip
    if not choosing:
        return False
    nodes = sorted(steps.chosen_node.unique())
    fig, axes = plt.subplots(
        1, len(choosing), figsize=(4.5 * len(choosing), 4), squeeze=False
    )
    for ax, label in zip(axes[0], choosing):
        d = steps[steps.strategy == label]
        share = pd.crosstab(d.step, d.chosen_node, normalize="index")
        bottom = np.zeros(len(share))
        for node in nodes:
            if node not in share:
                continue
            name = "observational" if node < 0 else f"X{node + 1}"
            ax.bar(share.index, share[node], bottom=bottom, label=name)
            bottom += share[node].to_numpy()
        ax.set_xlabel("step")
        ax.set_ylabel("share of chosen experiments")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7)
    _save(fig, prefix + "_chosen_node.png")
    return True


def plot_value_surrogate(
    trace: pd.DataFrame,
    surrogate: pd.DataFrame,
    scores: pd.DataFrame,
    prefix: str,
    *,
    task,
    seed,
    step: int,
    strategy: str,
    dense: pd.DataFrame | None = None,
) -> str | None:
    """Per node, the final GP posterior of EIG against the value at one
    (task, seed, step, strategy), with the evaluated points (error bars: two
    standard errors), the re-evaluated recommended value (star) and the
    dense diagnostic (grey) when given. Returns the file name or None."""

    def at(df):
        if df is None or df.empty:
            return df
        return df[
            (df.task == task) & (df.seed == seed) & (df.step == step)
            & (df.strategy == strategy)
        ]  # fmt: skip

    t, s, sc, dn = at(trace), at(surrogate), at(scores), at(dense)
    if t is None or t.empty:
        return None
    nodes = sorted(t.node.unique())
    fig, axes = plt.subplots(
        1, len(nodes), figsize=(4.5 * len(nodes), 4), squeeze=False
    )
    for ax, node in zip(axes[0], nodes):
        sn = s[s.node == node]
        ax.plot(sn.value, sn.posterior_mean, color="C0", label="GP mean")
        ax.fill_between(
            sn.value,
            sn.posterior_mean - 2 * sn.posterior_standard_deviation,
            sn.posterior_mean + 2 * sn.posterior_standard_deviation,
            color="C0", alpha=0.2, label="GP +- 2 sd",
        )  # fmt: skip
        tn = t[t.node == node]
        ax.errorbar(
            tn.value, tn.expected_information_gain, yerr=2 * tn.standard_error,
            fmt="o", ms=4, color="C3", label="evaluations",
        )  # fmt: skip
        if dn is not None and not dn.empty:
            dd = dn[dn.node == node]
            ax.plot(
                dd.value, dd.expected_information_gain, color="0.4", lw=1,
                ls="--", label="dense grid",
            )  # fmt: skip
        if sc is not None and not sc.empty:
            r = sc[sc.candidate_node == node]
            ax.plot(
                r.optimised_value, r.expected_information_gain, "*", ms=12,
                color="C2", label="re-evaluated",
            )  # fmt: skip
        ax.set_xlabel(f"do-value of X{node + 1}")
        ax.set_ylabel("EIG (nats)")
        ax.set_title(f"X{node + 1}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"task {task}, seed {seed}, step {step}, {strategy}", fontsize=9)
    path = f"{prefix}_surrogate_task{task}_seed{seed}_step{step}.png"
    _save(fig, path)
    return path


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def plot_active_learning_all(prefix: str, surrogate_step: int = 1) -> list[str]:
    """Every figure from the CSVs written with `prefix`; the surrogate figure
    is drawn for the first task and seed at `surrogate_step` (and for every
    step with a dense diagnostic). Returns the files written."""
    steps = _read(prefix + "_steps.csv")
    if steps.empty:
        return []
    written = []
    plot_kl_to_truth(steps, prefix)
    written.append(prefix + "_kl_to_truth.png")
    if plot_calibration(steps, prefix):
        written.append(prefix + "_calibration.png")
    if plot_chosen_node_frequency(steps, prefix):
        written.append(prefix + "_chosen_node.png")
    trace = _read(prefix + "_bayesian_optimisation_trace.csv")
    surrogate = _read(prefix + "_surrogate.csv")
    scores = _read(prefix + "_scores.csv")
    dense = _read(prefix + "_dense_value_diagnostic.csv")
    if trace.empty:
        return written
    strategy = trace.strategy.iloc[0]
    task, seed = trace.task.iloc[0], trace.seed.iloc[0]
    wanted = {surrogate_step}
    if not dense.empty:
        wanted |= set(dense[(dense.task == task) & (dense.seed == seed)].step)
    for step in sorted(wanted):
        path = plot_value_surrogate(
            trace, surrogate, scores, prefix, task=task, seed=seed,
            step=int(step), strategy=strategy, dense=dense,
        )  # fmt: skip
        if path:
            written.append(path)
    return written
