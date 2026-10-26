"""The six D=3 DAG motifs behind `graph_type: [MOTIF3]`.

Every labelled DAG on 3 nodes (25 in all) belongs to one class:

    empty      no edge                    1 labelled DAG
    single     one edge                   6
    chain      a -> b -> c                6
    fork       a <- b -> c                3
    collider   a -> c <- b                3
    complete   all three pairs adjacent   6 (transitive tournaments)

`MixedRegimeConfig.motif_probs` gives the class probabilities; a graph is then
uniform within its class (see `sample_graph`).
"""

from __future__ import annotations

import itertools

import numpy as np

MOTIF_CLASSES = ("empty", "single", "chain", "fork", "collider", "complete")
MOTIF_SIZES = {
    "empty": 1,
    "single": 6,
    "chain": 6,
    "fork": 3,
    "collider": 3,
    "complete": 6,
}


def motif_class(adj: np.ndarray) -> str:
    """Motif class (one of MOTIF_CLASSES) of a 3-node DAG, adj[i, j] = 1 iff i -> j."""
    a = np.asarray(adj) != 0
    if a.shape != (3, 3):
        raise ValueError(f"motifs are defined for D = 3, got shape {a.shape}")
    n_edges = int(a.sum())
    if n_edges == 0:
        return "empty"
    if n_edges == 1:
        return "single"
    if n_edges == 3:
        return "complete"
    if n_edges != 2:
        raise ValueError("not a DAG on 3 nodes")
    if a.sum(0).max() == 2:  # a node with two parents
        return "collider"
    if a.sum(1).max() == 2:  # a node with two children
        return "fork"
    return "chain"


def motif_dags() -> dict[str, list[np.ndarray]]:
    """Class -> list of all its labelled DAGs (int8 [3, 3]), in a fixed order."""
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    out: dict[str, list[np.ndarray]] = {c: [] for c in MOTIF_CLASSES}
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        adj = np.zeros((3, 3), dtype=np.int8)
        for (i, j), b in zip(pairs, bits):
            adj[i, j] = b
        # acyclic iff some node order has every edge pointing forward
        if any(
            all(
                adj[order[q], order[p]] == 0
                for p in range(3)
                for q in range(p, 3)
            )
            for order in itertools.permutations(range(3))
        ):
            out[motif_class(adj)].append(adj)
    return out


def validate_motif_probs(probs: dict[str, float]) -> dict[str, float]:
    """Check keys (subset of MOTIF_CLASSES), non-negativity and sum 1 (1e-9);
    returns the probabilities for all six classes (missing keys are 0)."""
    unknown = set(probs) - set(MOTIF_CLASSES)
    if unknown:
        raise ValueError(f"motif_probs: unknown classes {sorted(unknown)}")
    full = {c: float(probs.get(c, 0.0)) for c in MOTIF_CLASSES}
    if min(full.values()) < 0:
        raise ValueError("motif_probs must be non-negative")
    if abs(sum(full.values()) - 1.0) > 1e-9:
        raise ValueError(f"motif_probs must sum to 1, got {sum(full.values())}")
    return full
