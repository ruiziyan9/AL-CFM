"""
Statistics of tasks from the mixed-regime prior (numpy only, no plotting),
shared by the plot_* diagnostics in this package.

- path_lengths / relation: where an outcome sits relative to a treatment in
  the DAG (direct child, further downstream, upstream, confounded only, or
  independent).
- query_pair_records / context_pair_records: those relations for the
  (treatment, outcome) pairs the model is trained on. Outcomes are weighted
  uniformly over the D-1 non-treatment nodes, as the trainer draws them
  (mixed_context_trainer: Y uniform over {0..D-1} \\ {T}).
- edge_strengths: standardised regression coefficient of each child on its
  parents, estimated from observational rows. Exact in expectation for the
  linear families; a linear summary of the dependence for nonlinear ones.
- effect_gaps: how far do(X_t = x) moves X_y, compared with the marginal and
  with the observational conditional X_y | X_t ~= x.
- question_class / context_role: whether p(Y | T) and p(Y | do(T)) differ
  (an open back-door path from T to Y), and the role of a context
  intervention do(C) for a question (T, Y). Any D.
"""

from __future__ import annotations

import numpy as np

from CITNP.datasets.mixed_regime_generator import Task

__all__ = [
    "CONTEXT_ROLES",
    "DIFFER_CLASSES",
    "QUESTION_CLASSES",
    "RELATIONS",
    "context_role",
    "question_class",
    "conditional_rows",
    "context_pair_records",
    "edge_strengths",
    "effect_gaps",
    "path_lengths",
    "query_pair_records",
    "query_treatments",
    "relation",
]

RELATIONS = (
    "direct",  # t -> y
    "downstream_2",  # t -> . -> y (shortest directed path 2)
    "downstream_3+",
    "upstream",  # y is an ancestor of t: do(t) cannot move y
    "confounded",  # no directed path either way, common ancestor
    "independent",  # d-separated marginally
)


def path_lengths(graph: np.ndarray) -> np.ndarray:
    """dist[i, j] = length of the shortest directed path i -> j (0 on the
    diagonal, inf if none); graph[i, j] = 1 iff i -> j."""
    A = np.asarray(graph) != 0
    D = len(A)
    dist = np.full((D, D), np.inf)
    for s in range(D):
        dist[s, s] = 0
        frontier, d = [s], 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in np.flatnonzero(A[u]):
                    if dist[s, v] == np.inf:
                        dist[s, v] = d
                        nxt.append(int(v))
            frontier = nxt
    return dist


def relation(dist: np.ndarray, t: int, y: int) -> str:
    """One of RELATIONS for treatment t and outcome y (t != y)."""
    d = dist[t, y]
    if d == 1:
        return "direct"
    if d == 2:
        return "downstream_2"
    if np.isfinite(d):
        return "downstream_3+"
    if np.isfinite(dist[y, t]):
        return "upstream"
    if (np.isfinite(dist[:, t]) & np.isfinite(dist[:, y])).any():
        return "confounded"
    return "independent"


def query_treatments(task: Task) -> list[int]:
    """Treatment of each question: one per group in grouped mode, one per
    interventional query row in per-row mode (observational rows skipped)."""
    if task.meta.get("n_groups", 0):
        return [int(t) for t in task.meta["group_treatments"]]
    return [int(np.flatnonzero(r)[0]) for r in task.qry_int if r.any()]


def _pair_records(task: Task, treatments: list[int], weights: list[float], source: str):
    dist = path_lengths(task.graph)
    D = len(dist)
    total = float(sum(weights)) or 1.0
    out = []
    for t, w in zip(treatments, weights):
        for y in range(D):
            if y != t:
                out.append({
                    "source": source, "D": D, "t": t, "y": y,
                    "relation": relation(dist, t, y),
                    "path_len": float(dist[t, y]),
                    # sums to 1 per task: uniform outcome, questions weighted
                    "weight": w / total / (D - 1),
                })  # fmt: skip
    return out


def query_pair_records(task: Task) -> list[dict]:
    """(treatment, outcome) pairs of the task's questions."""
    ts = query_treatments(task)
    return _pair_records(task, ts, [1.0] * len(ts), "query")


def context_pair_records(task: Task) -> list[dict]:
    """(intervened node, other node) pairs of the context, weighted by the
    regime's number of rows."""
    ts, ws = [], []
    for r, targets in enumerate(task.regime_targets):
        for t in targets:
            ts.append(int(t))
            ws.append(float((task.ctx_regime == r).sum()))
    return _pair_records(task, ts, ws, "context")


def edge_strengths(x_obs: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """W[i, j] = coefficient of standardised X_i when regressing standardised
    X_j on all its parents (0 where there is no edge)."""
    z = (x_obs - x_obs.mean(0)) / (x_obs.std(0) + 1e-12)
    G = np.asarray(graph) != 0
    W = np.zeros(G.shape)
    for j in range(len(G)):
        pa = np.flatnonzero(G[:, j])
        if pa.size:
            coef, *_ = np.linalg.lstsq(z[:, pa], z[:, j], rcond=None)
            W[pa, j] = coef
    return W


def conditional_rows(x_t: np.ndarray, x: float, k: int) -> np.ndarray:
    """Indices of the k observational rows with X_t closest to x
    (nearest-neighbour estimate of X | X_t ~= x)."""
    return np.argsort(np.abs(x_t - x))[: min(k, len(x_t))]


def effect_gaps(
    x_obs: np.ndarray, x_int: np.ndarray, t: int, y: int, x: float, k: int
) -> dict:
    """Mean gaps in units of the observational sd of X_y:
    int_shift   = E[X_y | do(X_t=x)] - E[X_y]            (causal effect size)
    int_vs_cond = E[X_y | do(X_t=x)] - E[X_y | X_t ~= x]  (confounding bias)."""
    sd = x_obs[:, y].std() + 1e-12
    idx = conditional_rows(x_obs[:, t], x, k)
    m_int = x_int[:, y].mean()
    return {
        "int_shift": float((m_int - x_obs[:, y].mean()) / sd),
        "int_vs_cond": float((m_int - x_obs[idx, y].mean()) / sd),
        "cond_window": (float(x_obs[idx, t].min()), float(x_obs[idx, t].max())),
    }


# --------------------------------------------------------------------------- #
# Question classes: does conditioning answer the interventional question?
# --------------------------------------------------------------------------- #
QUESTION_CLASSES = (
    "reverse",  # Y is an ancestor of T
    "confounded effect",  # T -> ... -> Y plus a common cause (back-door path)
    "pure confounding",  # no directed path either way, common cause
    "unconfounded effect",  # T -> ... -> Y, no back-door path
    "unrelated",  # no directed path and no common cause
)
# p(Y | T=x) != p(Y | do(T=x)) generically iff there is an open back-door path.
DIFFER_CLASSES = frozenset(QUESTION_CLASSES[:3])

CONTEXT_ROLES = (
    "obs",
    "do(T)",  # samples the target distribution directly
    "do(confounder)",  # ancestor of T with a directed path to Y avoiding T
    "do(mediator)",  # on a directed path T -> ... -> Y
    "do(Y)",
    "do(other)",  # e.g. ancestors of T only, descendants of Y, unrelated
)


def _without(graph: np.ndarray, node: int) -> np.ndarray:
    g = np.array(graph, copy=True)
    g[node, :] = 0
    g[:, node] = 0
    return g


def question_class(graph: np.ndarray, t: int, y: int) -> str:
    """One of QUESTION_CLASSES for treatment t and outcome y (t != y).

    A back-door path from t to y with no conditioning is a trek
    t <- ... <- a -> ... -> y whose source a has a directed path to t and a
    directed path to y that avoids t (a = y is the reverse case)."""
    dist = path_lengths(graph)
    if np.isfinite(dist[y, t]):
        return "reverse"
    d_no_t = path_lengths(_without(graph, t))
    back = any(
        np.isfinite(dist[a, t]) and np.isfinite(d_no_t[a, y])
        for a in range(len(dist))
        if a not in (t, y)
    )
    if np.isfinite(dist[t, y]):
        return "confounded effect" if back else "unconfounded effect"
    return "pure confounding" if back else "unrelated"


def context_role(graph: np.ndarray, t: int, y: int, c: int | None) -> str:
    """Role (CONTEXT_ROLES) of a context regime do(X_c) for question (t, y);
    c=None is the observational regime."""
    if c is None:
        return "obs"
    if c == t:
        return "do(T)"
    if c == y:
        return "do(Y)"
    dist = path_lengths(graph)
    if np.isfinite(dist[t, c]) and np.isfinite(dist[c, y]):
        return "do(mediator)"
    if np.isfinite(dist[c, t]) and np.isfinite(path_lengths(_without(graph, t))[c, y]):
        return "do(confounder)"
    return "do(other)"
