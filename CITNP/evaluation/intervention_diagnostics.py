"""Do the model's hidden states keep track of which context rows are
intervened? Encoding-agnostic diagnostics on a trained model.

`run_intervention_probes` fits linear probes on the per-layer context states
(`model_api.context_states`) of tasks drawn from the training prior. It works
for every intervention encoding; the two probes on the extra-node metadata
column run only for extra-node models (`model_api.uses_metadata_node`).

`run_context_edits` tests label sensitivity: it edits only the intervention
mask of a context (values untouched), so the edited context is deliberately
mislabelled, and compares how much the model's prediction moves with how
much the prediction of the exact Bayesian oracle given the SAME edited
context moves (`oracle.LinGaussOracle`: a reasoner who believes the labels;
linear-Gaussian priors with normalise: false, D <= 4). It does not ask
whether either recovers the truth. `summarise_context_edits` turns the
per-query rows into the model-vs-oracle sensitivity tables and curves.

Like `experiments.py`, every `run_*` takes a loaded model (see
`model_api.load_model`) and returns plain results (pandas DataFrames); writing
files and printing summaries are left to the caller.
"""

from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from CITNP.datasets.linear_gaussian_prior import LinearGaussianPrior
from CITNP.evaluation.experiments import _value_features
from CITNP.evaluation.metrics import (
    kl_mc,
    mix_logpdf,
    mixture_moments,
    oracle_mixture_on_grid,
    oracle_weights,
    sample_mixture,
    sample_oracle,
)
from CITNP.evaluation.model_api import (
    context_states,
    predict_mixture,
    probe_points,
    uses_metadata_node,
)
from CITNP.evaluation.oracle import LinGaussOracle
from CITNP.evaluation.probes import fit_classification_probe, shuffle_labels
from CITNP.evaluation.tasks import (
    empirical_graph_prior,
    linear_scm_from_debug,
    observationally_identified,
    prior_task,
    require_exact_oracle,
    resolve_graph_prior_samples,
)

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


def query_rows_by_treatment(task) -> dict[int, np.ndarray]:
    """Treatment T -> indices of the task's query rows that intervene on T
    only, for every T that has such rows (in increasing T)."""
    query_int = np.asarray(task.qry_int, dtype=bool)
    single = query_int.sum(1) == 1
    out = {}
    for T in range(query_int.shape[1]):
        rows = np.flatnonzero(single & query_int[:, T])
        if len(rows):
            out[T] = rows
    return out


def query_treatments(task) -> list[int]:
    """Treatments T of the task's query rows (rows with exactly one
    intervened variable), sorted and distinct."""
    return list(query_rows_by_treatment(task))


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


# --------------------------------------------------------------------------- #
# Label sensitivity: counterfactual edits of the intervention mask
# --------------------------------------------------------------------------- #
CONTEXT_EDITS = {
    "relabel_observational": (
        "the first n intervened rows (seeded order) relabelled observational"
    ),
    "move_target": (
        "the first n intervened rows (seeded order) have their intervention "
        "moved from variable j to a seeded variable k != j"
    ),
}
# Block size of `metrics.mix_logpdf` over oracle mixture components.
_MIXTURE_CHUNK = 8192


def edit_order(intervened: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The intervened rows of a mask [N, D] in a random order: edits with n
    rows edit the first n of them, so the edits are nested in n."""
    return rng.permutation(np.flatnonzero(np.asarray(intervened).any(1)))


def edit_counts(num_intervened: int, max_steps: int | None = None) -> list[int]:
    """Numbers n of edited rows: every n = 1 .. num_intervened, or, with
    max_steps, at most max_steps of them spaced geometrically (1 and
    num_intervened always included)."""
    if max_steps is None or max_steps >= num_intervened:
        return list(range(1, num_intervened + 1))
    grid = np.geomspace(1, num_intervened, max(int(max_steps), 2))
    return sorted({int(round(n)) for n in grid} | {1, num_intervened})


def relabel_rows_as_observational(
    intervened: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Edit 1: a copy of the mask [N, D] with `rows` relabelled
    observational (all their cells False)."""
    edited = np.array(intervened, dtype=bool)
    edited[np.asarray(rows, dtype=int)] = False
    return edited


def draw_new_targets(
    intervened: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """For every row of a mask [N, D]: (moved_from, moved_to) [N] int, the
    intervened variable j to move and the variable k it moves to, drawn
    with `rng` (j uniform over the row's targets, k uniform over the
    variables the row does not intervene on). -1 for rows that are
    observational or intervene on every variable (nothing to move)."""
    intervened = np.asarray(intervened, dtype=bool)
    N, D = intervened.shape
    moved_from = np.full(N, -1)
    moved_to = np.full(N, -1)
    for r in range(N):
        targets = np.flatnonzero(intervened[r])
        free = np.flatnonzero(~intervened[r])
        if len(targets) and len(free):
            moved_from[r] = rng.choice(targets)
            moved_to[r] = rng.choice(free)
    return moved_from, moved_to


def move_row_targets(
    intervened: np.ndarray,
    rows: np.ndarray,
    moved_from: np.ndarray,
    moved_to: np.ndarray,
) -> np.ndarray:
    """Edit 2: a copy of the mask [N, D] in which each row r of `rows` has
    its intervention on moved_from[r] moved to moved_to[r] (rows with -1
    are left unchanged). Values are not touched, so a moved row's do-value
    is now the value its new target happened to have."""
    edited = np.array(intervened, dtype=bool)
    for r in np.asarray(rows, dtype=int):
        if moved_from[r] >= 0:
            edited[r, moved_from[r]] = False
            edited[r, moved_to[r]] = True
    return edited


def _slope(t: np.ndarray, mean: np.ndarray) -> float:
    """Least-squares slope of mean(t); NaN for fewer than two distinct t."""
    if len(np.unique(t)) < 2:
        return float("nan")
    return float(np.polyfit(t, mean, 1)[0])


def _correlation(a, b) -> float:
    """Pearson correlation; NaN for fewer than two values or no spread."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _sign_agreement(model_shift, oracle_shift, tol: float = 1e-9) -> float:
    """Fraction of entries where the model's shift has the sign of the
    oracle's, among entries where the oracle's shift is not ~0; NaN if
    there are none."""
    model_shift = np.asarray(model_shift, dtype=float)
    oracle_shift = np.asarray(oracle_shift, dtype=float)
    finite = np.isfinite(model_shift) & np.isfinite(oracle_shift)
    if not finite.any():
        return float("nan")
    scale = max(1.0, float(np.abs(oracle_shift[finite]).max()))
    keep = finite & (np.abs(np.where(finite, oracle_shift, 0.0)) > tol * scale)
    if not keep.any():
        return float("nan")
    same = np.sign(model_shift[keep]) == np.sign(oracle_shift[keep])
    return float(same.mean())


def _predictives(predict, oracle, x, mask, queries, normalisation_int_mask,
                 oracle_kw):  # fmt: skip
    """Model and oracle predictive distributions for every query of one
    (possibly edited) context.

    queries: T -> do-values t [n]. Returns {(T, Y): {"model": (mu, sd, logw)
    float64 CPU tensors [n, K], "oracle": (log_w, w, slope, var) arrays
    [M], "model_mean", "oracle_mean": [n]}}.
    """
    D = x.shape[1]
    out = {}
    for T, t in queries.items():
        Ys = np.array([Y for Y in range(D) if Y != T])
        mu, sd, logw = predict(
            x, mask, np.full(len(Ys), T), Ys, t,
            normalisation_int_mask=normalisation_int_mask,
        )  # fmt: skip
        mu, sd, logw = mu.cpu(), sd.cpu(), logw.cpu()
        for q, Y in enumerate(Ys.tolist()):
            model = (mu[q], sd[q], logw[q])
            model_mean, _ = mixture_moments(*(p.numpy() for p in model))
            log_w, slope, var = oracle.predictive_components(T, Y, **oracle_kw)
            log_w, w, var = oracle_weights(log_w, var)
            out[(T, Y)] = {
                "model": model, "oracle": (log_w, w, slope, var),
                "model_mean": model_mean, "oracle_mean": float(w @ slope) * t,
            }  # fmt: skip
    return out


def _oracle_logpdf(y: torch.Tensor, oracle_parameters, t: torch.Tensor):
    """log density [n, n_y] of y under the oracle mixture at do-values t."""
    log_w, _, slope, var = oracle_parameters
    mixture = oracle_mixture_on_grid(log_w, slope, var, t)
    return mix_logpdf(y, *mixture, chunk=_MIXTURE_CHUNK)


def run_context_edits(
    predict_or_model,
    data_cfg,
    *,
    D: int,
    device,
    n_tasks: int = 200,
    task_offset: int = 0,
    base_seed: int = 9301,
    n_mc: int = 2000,
    n_y: int = 64,
    min_graph_prob: float = 1e-4,
    max_query_points: int = 32,
    max_edit_steps: int | None = None,
    seed: int = 0,
    normalise: bool = True,
    keep_original_normalisation: bool = False,
    max_oracle_nodes: int = 4,
    log_graph_prior: dict | None = None,
    graph_prior_samples: int | None = None,
) -> pd.DataFrame:
    """Label sensitivity: how do the model's and the exact oracle's
    predictions change when only the intervention mask of the context is
    edited, as a function of how many rows are edited?

    Tasks task_offset .. task_offset + n_tasks - 1 of the training prior
    (`tasks.prior_task(data_cfg, i, base_seed, D)`); tasks whose context has
    no intervened row are skipped. Per task, default_rng([seed, i]) draws a
    random order of the intervened rows (`edit_order`) and, for every row, a
    new target (`draw_new_targets`). For each edit of `CONTEXT_EDITS` and
    n in `edit_counts(number of intervened rows, max_edit_steps)` (by default
    every n from 1 to all), the first n rows of the order are edited:
        relabel_observational   their cells set to False
        move_target             their intervention moved j -> k != j
    The values x never change, and the edited mask goes through the model's
    normal input path (context_int_mask). Any variable may be the new
    target, the query's treatment and outcome included: neither the model
    nor the oracle needs a restriction.

    Queries: for each treatment T of the task's query rows
    (`query_rows_by_treatment`), the do-values t of its first
    max_query_points rows, and every outcome Y != T. The oracle is refitted
    on (x, edited mask) and the model rerun on every edited context. Per
    query, edit and n, for model and oracle alike (each against its own
    prediction on the unedited context):
        *_mean_shift            mean over t of (edited - original)
                                predictive mean
        *_mean_shift_rms        root mean square over t of that shift
        *_slope_change          change of the predictive mean's slope in t
        *_kl_original_to_edited KL(original || edited) predictive, Monte
                                Carlo with n_y draws per t from the original
                                (the same draws for every n), mean over t,
                                as the other evaluations compare mixtures
                                (`metrics.kl_mc`)
        mean_shift_sign_agreement
                                fraction of query points where the model's
                                mean shift has the oracle's sign
    Secondary only: secondary_kl_oracle_to_model_original / _edited, the
    usual KL(oracle || model) on the unedited and the edited context. The
    edited contexts are mislabelled, so they are outside the training
    distribution: a large discrepancy there is expected and is not a
    failure of the model on real data.

    Strata columns: identifiable_from_observational
    (`tasks.observationally_identified` on the task's true SCM),
    treatment_intervened_in_context and outcome_intervened_in_context (any
    intervened cell in column T / Y of the unedited context), relation
    (`question_class` of the true graph), and
    num_edited_rows_targeting_treatment (edited rows that intervened on T).

    The model standardises each context with its observational rows, so
    relabelling rows observational also changes the standardisation, as it
    would for a context that really was labelled that way (the oracle sees
    the raw values). keep_original_normalisation=True instead keeps the
    unedited context's standardisation for every edit (an ablation).
    move_target never changes it: observational rows stay observational.
    The oracle's MC draws use seed i for every context version.

    Args:
    ----------
    predict_or_model : torch.nn.Module | callable
        A loaded model (any size and intervention encoding), or a predictor
        (x, m, Ts, Ys, t, normalisation_int_mask=None) -> (mu, sd, logw)
        [Q, n, K] on the raw scale, like `model_api.predict_mixture`.

    data_cfg : MixedRegimeConfig
        Linear-Gaussian prior with normalise: false (`require_exact_oracle`).

    D : int
        Number of variables (at most max_oracle_nodes).

    n_mc, n_y, min_graph_prob : int, int, float
        Oracle MC parameter draws per query (budget "total"), MC draws per
        do-value for the KLs, DAG posterior cut-off.

    max_edit_steps : int | None
        Cap on the number of n values per task and edit (cost grows
        linearly in it); None: every n.

    log_graph_prior, graph_prior_samples
        Oracle graph prior, as in `experiments.run_kshot_prior`.

    Returns:
    ----------
    One row per (task, T, Y, edit, n); see `summarise_context_edits`.
    """
    from CITNP.prior_diagnostics.stats import question_class

    if D > max_oracle_nodes:
        raise ValueError(
            f"D={D} > max_oracle_nodes={max_oracle_nodes}: exact DAG "
            "enumeration is infeasible"
        )
    require_exact_oracle(data_cfg)
    if isinstance(predict_or_model, torch.nn.Module):
        predict = partial(
            predict_mixture, predict_or_model, device=device, normalise=normalise
        )
    else:
        predict = predict_or_model
    graph_prior = log_graph_prior
    if graph_prior is None:
        graph_prior = empirical_graph_prior(
            data_cfg,
            n=resolve_graph_prior_samples(D, graph_prior_samples),
            num_nodes=D,
        )
    parameter_prior = LinearGaussianPrior.from_config(data_cfg)

    rows: list[dict] = []
    num_skipped = 0
    for i in range(task_offset, task_offset + n_tasks):
        task, debug = prior_task(data_cfg, i, base_seed, D, return_debug=True)
        scm = linear_scm_from_debug(debug)
        graph = task.graph.astype(int)
        x = np.asarray(task.ctx_x, dtype=np.float64)
        original = np.asarray(task.ctx_int, dtype=bool)
        if not original.any():
            num_skipped += 1
            continue
        rng = np.random.default_rng([seed, i])
        order = edit_order(original, rng)
        moved_from, moved_to = draw_new_targets(original, rng)
        query_x = np.asarray(task.qry_x, dtype=np.float64)
        queries = {
            T: query_x[query_rows[:max_query_points], T]
            for T, query_rows in query_rows_by_treatment(task).items()
        }
        oracle_kw = dict(
            n_mc=n_mc, min_graph_prob=min_graph_prob, seed=i, budget="total"
        )

        def predictives(mask):
            oracle = LinGaussOracle(
                D, log_graph_prior=graph_prior, prior=parameter_prior
            ).fit(x, mask)
            return _predictives(
                predict, oracle, x, mask, queries,
                original if keep_original_normalisation else None,
                {**oracle_kw, "param_cache": {}},
            )  # fmt: skip

        base = predictives(original)
        # per query: draws from the original predictives and their log
        # densities, shared by every edit and n
        reference = {}
        for (T, Y), p in base.items():
            t = torch.as_tensor(queries[T], dtype=torch.float64)
            qrng = np.random.default_rng([seed, i, T, Y])
            mu, sd, logw = p["model"]
            y_model = torch.as_tensor(
                sample_mixture(mu.numpy(), sd.numpy(), logw.numpy(), n_y, qrng)
            )
            _, w, slope, var = p["oracle"]
            y_oracle = torch.as_tensor(
                sample_oracle(w, slope, var, queries[T], n_y, qrng)
            )
            lp_oracle = _oracle_logpdf(y_oracle, p["oracle"], t)
            reference[(T, Y)] = {
                "t": t,
                "y_model": y_model,
                "lp_model": mix_logpdf(y_model, *p["model"]),
                "y_oracle": y_oracle,
                "lp_oracle": lp_oracle,
                "kl_oracle_model": float(
                    kl_mc(lp_oracle, mix_logpdf(y_oracle, *p["model"])).mean()
                ),
            }

        counts = edit_counts(len(order), max_edit_steps)
        for edit_index, edit in enumerate(CONTEXT_EDITS):
            for n in counts:
                edited_rows = order[:n]
                if edit == "relabel_observational":
                    mask = relabel_rows_as_observational(original, edited_rows)
                else:
                    mask = move_row_targets(
                        original, edited_rows, moved_from, moved_to
                    )
                after = predictives(mask)
                for (T, Y), before in base.items():
                    ref = reference[(T, Y)]
                    now = after[(T, Y)]
                    t_values = queries[T]
                    shift = {
                        who: now[f"{who}_mean"] - before[f"{who}_mean"]
                        for who in ("model", "oracle")
                    }
                    kl_model = kl_mc(
                        ref["lp_model"], mix_logpdf(ref["y_model"], *now["model"])
                    )
                    kl_oracle = kl_mc(
                        ref["lp_oracle"],
                        _oracle_logpdf(ref["y_oracle"], now["oracle"], ref["t"]),
                    )
                    srng = np.random.default_rng([seed, i, T, Y, edit_index, n])
                    _, w, slope, var = now["oracle"]
                    y_edit = torch.as_tensor(
                        sample_oracle(w, slope, var, t_values, n_y, srng)
                    )
                    kl_secondary = kl_mc(
                        _oracle_logpdf(y_edit, now["oracle"], ref["t"]),
                        mix_logpdf(y_edit, *now["model"]),
                    )
                    row = {
                        "D": D, "task": i, "T": T, "Y": Y, "edit": edit,
                        "num_edited_rows": n,
                        "num_intervened_rows": len(order),
                        "fraction_edited_rows": n / len(order),
                        "num_edited_rows_targeting_treatment": int(
                            original[edited_rows, T].sum()
                        ),
                        "relation": question_class(graph, T, Y),
                        "identifiable_from_observational":
                            observationally_identified(scm, T, Y),
                        "treatment_intervened_in_context":
                            bool(original[:, T].any()),
                        "outcome_intervened_in_context":
                            bool(original[:, Y].any()),
                        "num_query_points": len(t_values),
                    }  # fmt: skip
                    for who in ("model", "oracle"):
                        row[f"{who}_mean_shift"] = float(shift[who].mean())
                        row[f"{who}_mean_shift_rms"] = float(
                            np.sqrt(np.mean(shift[who] ** 2))
                        )
                        row[f"{who}_slope_change"] = _slope(t_values, shift[who])
                    row["model_kl_original_to_edited"] = float(kl_model.mean())
                    row["oracle_kl_original_to_edited"] = float(kl_oracle.mean())
                    row["mean_shift_sign_agreement"] = _sign_agreement(
                        shift["model"], shift["oracle"]
                    )
                    row["secondary_kl_oracle_to_model_original"] = ref[
                        "kl_oracle_model"
                    ]
                    row["secondary_kl_oracle_to_model_edited"] = float(
                        kl_secondary.mean()
                    )
                    rows.append(row)
        if (i - task_offset + 1) % 10 == 0:
            print(f"context edits: {i - task_offset + 1}/{n_tasks} tasks")
    if num_skipped:
        print(f"context edits: skipped {num_skipped} task(s) without an "
              "intervened context row")  # fmt: skip
    return pd.DataFrame(rows)


CONTEXT_EDIT_STRATA = (
    "identifiable_from_observational",
    "treatment_intervened_in_context",
)
ALL = "all"


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def _summary_row(group: pd.DataFrame) -> dict:
    """Model-vs-oracle sensitivity statistics of one group of rows."""
    model_rms = group["model_mean_shift_rms"].mean()
    oracle_rms = group["oracle_mean_shift_rms"].mean()
    model_kl = group["model_kl_original_to_edited"].mean()
    oracle_kl = group["oracle_kl_original_to_edited"].mean()
    model_slope = group["model_slope_change"].abs().mean()
    oracle_slope = group["oracle_slope_change"].abs().mean()
    return {
        "num_queries": len(group),
        "num_tasks": group["task"].nunique(),
        "model_mean_shift_rms": model_rms,
        "oracle_mean_shift_rms": oracle_rms,
        "mean_shift_ratio": _ratio(model_rms, oracle_rms),
        "model_abs_slope_change": model_slope,
        "oracle_abs_slope_change": oracle_slope,
        "slope_change_ratio": _ratio(model_slope, oracle_slope),
        "model_kl_original_to_edited": model_kl,
        "oracle_kl_original_to_edited": oracle_kl,
        "kl_ratio": _ratio(model_kl, oracle_kl),
        "mean_shift_correlation": _correlation(
            group["model_mean_shift"], group["oracle_mean_shift"]
        ),
        "slope_change_correlation": _correlation(
            group["model_slope_change"], group["oracle_slope_change"]
        ),
        "kl_correlation": _correlation(
            group["model_kl_original_to_edited"],
            group["oracle_kl_original_to_edited"],
        ),
        "mean_shift_sign_agreement": group["mean_shift_sign_agreement"].mean(),
        "slope_change_sign_agreement": _sign_agreement(
            group["model_slope_change"], group["oracle_slope_change"]
        ),
        "secondary_kl_oracle_to_model_original": group[
            "secondary_kl_oracle_to_model_original"
        ].mean(),
        "secondary_kl_oracle_to_model_edited": group[
            "secondary_kl_oracle_to_model_edited"
        ].mean(),
    }


def summarise_context_edits(df: pd.DataFrame) -> pd.DataFrame:
    """Model-vs-oracle label sensitivity per edit, overall, per stratum and
    per number of edited rows.

    Groups the rows of `run_context_edits` by edit, by every subset of
    `CONTEXT_EDIT_STRATA`, and by num_edited_rows or not; a key column holds
    "all" where it is not grouped on. The rows with num_edited_rows != "all"
    form the shift-vs-number-of-edited-rows curves. Columns per group:
        num_queries, num_tasks
        model_ / oracle_mean_shift_rms, mean_shift_ratio (model / oracle)
        model_ / oracle_abs_slope_change, slope_change_ratio
        model_ / oracle_kl_original_to_edited, kl_ratio
        mean_shift_correlation, slope_change_correlation, kl_correlation
                                  Pearson correlation over queries
        mean_shift_sign_agreement mean over queries of the per-point value
        slope_change_sign_agreement
                                  fraction of queries whose slope changes
                                  have the same sign
        secondary_kl_oracle_to_model_original / _edited   (see
                                  `run_context_edits`: secondary only)
    A ratio near 1 with high correlation and sign agreement means the model
    reacts to the labels as much as, and where, a Bayesian reasoner who
    believes them does; a ratio near 0 means it ignores them.
    """
    if df.empty:
        return pd.DataFrame()
    keys_first = ["edit", *CONTEXT_EDIT_STRATA, "num_edited_rows"]
    out = []
    for by_count in (False, True):
        for size in range(len(CONTEXT_EDIT_STRATA) + 1):
            for subset in combinations(CONTEXT_EDIT_STRATA, size):
                keys = ["edit", *subset]
                if by_count:
                    keys.append("num_edited_rows")
                for values, group in df.groupby(keys):
                    values = values if isinstance(values, tuple) else (values,)
                    row: dict = {k: ALL for k in keys_first}
                    row.update(dict(zip(keys, values)))
                    row.update(_summary_row(group))
                    out.append(row)
    summary = pd.DataFrame(out)
    rest = [c for c in summary.columns if c not in keys_first]
    return summary[keys_first + rest]
