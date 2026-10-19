"""Deprecated: use `CITNP.evaluation.oracle` (re-exported here).

`make_context` still lives here until it moves to `CITNP.evaluation.tasks`;
the DeprecationWarning is added once nothing in the package imports this
module any more.
"""

import numpy as np

from CITNP.evaluation.oracle import (  # noqa: F401  (re-exports)
    ETA,
    NODE_A,
    NODE_B,
    ROOT_A,
    ROOT_B,
    LinearSCM,
    LinGaussOracle,
    NodePosterior,
    all_dags,
    dag_key,
    has_path,
    is_acyclic,
    node_posterior,
    topo,
)


def make_context(scm: LinearSCM, regimes: list[tuple[int | None, int]], do_scale: float, rng):
    """regimes: list of (intervened node or None for obs, n_rows). Returns x, mask."""
    xs, ms = [], []
    for node, n in regimes:
        if n == 0:
            continue
        dv = rng.normal(0, do_scale, n) if node is not None else None
        x, m = scm.sample(n, node, dv, rng)
        xs.append(x)
        ms.append(m)
    x, m = np.concatenate(xs), np.concatenate(ms)
    perm = rng.permutation(len(x))
    return x[perm], m[perm]
