"""Do the model's hidden states keep track of which context rows are
intervened? Encoding-agnostic diagnostics on a trained model.

`run_intervention_probes` fits linear probes on the per-layer context states
(`model_api.context_states`) of tasks drawn from the training prior. It works
for every intervention encoding; the two probes on the extra-node metadata
column run only for extra-node models (`model_api.uses_metadata_node`).

Like `experiments.py`, every `run_*` takes a loaded model (see
`model_api.load_model`) and returns plain results (pandas DataFrames); writing
files and printing summaries are left to the caller.
"""

import numpy as np
import pandas as pd
import torch

from CITNP.evaluation.experiments import _value_features
from CITNP.evaluation.model_api import (
    context_states,
    probe_points,
    uses_metadata_node,
)
from CITNP.evaluation.probes import fit_classification_probe, shuffle_labels
from CITNP.evaluation.tasks import prior_task

# Probe name -> what it predicts, from which states. "metadata_*" probes need
# the extra-node metadata column; "cell_*" probes use the D data cells.
INTERVENTION_PROBES = {
    "metadata_row_intervened": (
        "metadata column: is the row intervened (any variable) or "
        "observational?"
    ),
    "metadata_intervention_target": (
        "metadata column of single-target intervened rows: which variable "
        "was intervened? (D classes)"
    ),
    "cell_row_intervened": (
        "data cell: does its row contain an intervened cell?"
    ),
    "cell_row_intervened_excluding_targets": (
        "data cell that is NOT itself intervened: does its row contain an "
        "intervened cell? (row information that reached the other cells)"
    ),
    "cell_is_intervention_target": (
        "data cell: is this cell itself intervened?"
    ),
}
METADATA_PROBES = tuple(
    p for p in INTERVENTION_PROBES if p.startswith("metadata")
)
CELL_PROBES = tuple(p for p in INTERVENTION_PROBES if p.startswith("cell"))
VALUE_BASELINE_POINT = "value_baseline"


def query_treatments(task) -> list[int]:
    """Treatments T of the task's query rows (rows with exactly one
    intervened variable), sorted and distinct."""
    single = np.asarray(task.qry_int, dtype=bool)
    single = single[single.sum(1) == 1]
    return sorted({int(np.argmax(row)) for row in single})


def choose_question(task, rng: np.random.Generator) -> tuple[int, int]:
    """(T, Y) for the role embedding: T one of the task's query treatments
    (any variable if there are none), Y a different variable, both drawn
    with `rng`."""
    D = task.ctx_x.shape[1]
    treatments = query_treatments(task) or list(range(D))
    T = int(rng.choice(treatments))
    Y = int(rng.choice([j for j in range(D) if j != T]))
    return T, Y


def pick_rows(
    intervened: np.ndarray, num_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """Up to num_rows sorted row indices of a context mask [N, D]: half of
    them rows with an intervened cell where possible, the rest
    observational."""
    row_intervened = np.asarray(intervened, dtype=bool).any(1)
    positive = np.flatnonzero(row_intervened)
    negative = np.flatnonzero(~row_intervened)
    num_positive = min(num_rows // 2, len(positive))
    num_negative = min(num_rows - num_positive, len(negative))
    num_positive = min(num_rows - num_negative, len(positive))
    return np.sort(
        np.concatenate([
            rng.choice(positive, num_positive, replace=False),
            rng.choice(negative, num_negative, replace=False),
        ])
    )  # fmt: skip


def _probe_samples(
    states: dict[str, torch.Tensor],
    intervened: np.ndarray,
    rows: np.ndarray,
    num_variables: int,
    with_metadata: bool,
) -> dict[str, tuple[dict[str, np.ndarray], np.ndarray]]:
    """Features per probe point and labels of every probe, for the chosen
    rows of one context.

    Returns {probe: ({point: features [n, d_model]}, labels [n] int)}.
    """
    D = num_variables
    mask = np.asarray(intervened, dtype=bool)[rows]  # [R, D]
    row_intervened = mask.any(1)  # [R]
    cell_row_intervened = np.repeat(row_intervened, D)  # [R * D]
    cell_is_target = mask.ravel()
    single_target = mask.sum(1) == 1

    def cells(select=None):
        out = {}
        for name, value in states.items():
            x = value[rows, :D].reshape(len(rows) * D, -1).numpy()
            out[name] = x if select is None else x[select]
        return out

    samples = {
        "cell_row_intervened": (cells(), cell_row_intervened.astype(np.int64)),
        "cell_row_intervened_excluding_targets": (
            cells(~cell_is_target),
            cell_row_intervened[~cell_is_target].astype(np.int64),
        ),
        "cell_is_intervention_target": (
            cells(),
            cell_is_target.astype(np.int64),
        ),
    }
    if with_metadata:
        metadata = {name: v[rows, D].numpy() for name, v in states.items()}
        samples["metadata_row_intervened"] = (
            metadata,
            row_intervened.astype(np.int64),
        )
        samples["metadata_intervention_target"] = (
            {name: x[single_target] for name, x in metadata.items()},
            np.argmax(mask[single_target], axis=1).astype(np.int64)
            if single_target.any()
            else np.zeros(0, dtype=np.int64),
        )
    return samples


def _value_baseline_samples(
    ctx_x: np.ndarray, intervened: np.ndarray, rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Baseline features of the cell probes (`experiments._value_features`:
    the cell's value standardised by its column's non-intervened cells, plus
    a one-hot of the variable), in the sample order of `_probe_samples`."""
    N, D = np.asarray(intervened).shape
    features = _value_features(ctx_x, np.asarray(intervened, dtype=bool))
    features = features.reshape(N, D, -1)[rows].reshape(len(rows) * D, -1)
    is_target = np.asarray(intervened, dtype=bool)[rows].ravel()
    return {
        "cell_row_intervened": features,
        "cell_row_intervened_excluding_targets": features[~is_target],
        "cell_is_intervention_target": features,
    }


def _fit_or_missing(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, **probe
) -> dict:
    """`fit_classification_probe`, or NaN metrics when the probe cannot be
    fitted (one class only, or fewer tasks than folds)."""
    num_groups = len(np.unique(groups))
    if len(labels) == 0 or len(np.unique(labels)) < 2 or (
        num_groups < probe["n_folds"]
    ):
        nan = float("nan")
        return dict(
            accuracy=nan, balanced_accuracy=nan, auc=nan,
            majority_class_rate=nan, n=int(len(labels)),
            num_classes=int(len(np.unique(labels))), num_groups=num_groups,
        )  # fmt: skip
    return fit_classification_probe(features, labels, groups, **probe)


def run_intervention_probes(
    model: torch.nn.Module,
    data_cfg,
    *,
    D: int,
    device,
    n_tasks: int = 200,
    task_offset: int = 0,
    base_seed: int = 9201,
    rows_per_task: int = 32,
    n_folds: int = 5,
    l2: float = 1e-3,
    max_iter: int = 200,
    seed: int = 0,
    normalise: bool = True,
) -> pd.DataFrame:
    """Per-layer linear probes: is "this row / this cell is intervened" still
    linearly decodable from the context states at each depth?

    Tasks task_offset .. task_offset + n_tasks - 1 of the training prior
    (`tasks.prior_task(data_cfg, i, base_seed, D)`). Per task, rows_per_task
    context rows (half with an intervened cell where possible, `pick_rows`)
    and one question (T, Y) for the role embedding (`choose_question`), both
    drawn from default_rng([seed, i]). The states are those of
    `model_api.context_states` at every `probe_points(model)` depth. Probes
    (`INTERVENTION_PROBES`):
        metadata_row_intervened            (extra-node models only)
        metadata_intervention_target       (extra-node models only)
        cell_row_intervened
        cell_row_intervened_excluding_targets
        cell_is_intervention_target
    Each probe is class-balanced softmax regression
    (`probes.fit_classification_probe`) scored out of fold, with folds split
    by TASK (all rows of a task are in the same fold, never split by row).
    Every probe is also fitted on shuffled labels (`probes.shuffle_labels`,
    one permutation of all the probe's labels with seed [seed, probe
    index], shared by all depths): the shuffled-label control, at chance up
    to the probe's overfitting. Cell probes also get a "value_baseline" row
    (depth -1): the cell's standardised value and variable one-hot only.

    Args:
    ----------
    model : torch.nn.Module
        Any mixed-context model (any size and intervention encoding), in eval
        mode.

    data_cfg : MixedRegimeConfig
        The prior to draw tasks from (any function family).

    D : int
        Number of variables.

    rows_per_task : int
        Context rows per task used by the probes (each gives one metadata
        sample and D cell samples).

    n_folds, l2, max_iter : int, float, int
        Probe cross-validation folds and fitting settings.

    seed : int
        Seed of the row / question choice, the folds and the shuffles; the
        result is deterministic given the arguments.

    normalise : bool
        Per-task standardisation of the context, as in training.

    Returns:
    ----------
    One row per (probe, point, labels) with columns probe, point, depth
    (index in probe_points; -1 for the value baseline), labels ("true" or
    "shuffled"), accuracy, balanced_accuracy, auc (binary probes),
    majority_class_rate, n, num_classes, num_groups (tasks with samples).
    """
    names = probe_points(model)
    with_metadata = uses_metadata_node(model)
    probes = CELL_PROBES + (METADATA_PROBES if with_metadata else ())
    features: dict = {(p, n): [] for p in probes for n in names}
    baseline: dict = {p: [] for p in CELL_PROBES}
    labels: dict = {p: [] for p in probes}
    groups: dict = {p: [] for p in probes}
    for i in range(task_offset, task_offset + n_tasks):
        task = prior_task(data_cfg, i, base_seed, D)
        rng = np.random.default_rng([seed, i])
        intervened = np.asarray(task.ctx_int, dtype=bool)
        T, Y = choose_question(task, rng)
        rows = pick_rows(intervened, rows_per_task, rng)
        states = context_states(
            model, task.ctx_x, intervened, T, Y, device=device,
            normalise=normalise, include_metadata_node=with_metadata,
        )  # fmt: skip
        samples = _probe_samples(states, intervened, rows, D, with_metadata)
        base = _value_baseline_samples(task.ctx_x, intervened, rows)
        for p in probes:
            point_features, probe_labels = samples[p]
            for n in names:
                features[(p, n)].append(point_features[n])
            labels[p].append(probe_labels)
            groups[p].append(np.full(len(probe_labels), i))
            if p in baseline:
                baseline[p].append(base[p])
        if (i - task_offset + 1) % 50 == 0:
            print(f"probe states: {i - task_offset + 1}/{n_tasks} tasks")

    settings = dict(n_folds=n_folds, l2=l2, max_iter=max_iter, seed=seed)
    rows_out: list[dict] = []
    for p_index, p in enumerate(probes):
        y = np.concatenate(labels[p])
        g = np.concatenate(groups[p])
        variants = {"true": y, "shuffled": shuffle_labels(y, [seed, p_index])}
        points = [(VALUE_BASELINE_POINT, -1)] if p in baseline else []
        points += [(n, depth) for depth, n in enumerate(names)]
        for point, depth in points:
            if point == VALUE_BASELINE_POINT:
                X = np.concatenate(baseline[p])
            else:
                X = np.concatenate(features[(p, point)])
            for variant, y_variant in variants.items():
                rows_out.append({
                    "probe": p, "point": point, "depth": depth,
                    "labels": variant,
                    **_fit_or_missing(X, y_variant, g, **settings),
                })  # fmt: skip
    return pd.DataFrame(rows_out)
