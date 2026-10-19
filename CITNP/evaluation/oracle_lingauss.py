"""Deprecated: use `CITNP.evaluation.oracle` (oracle, SCMs) and
`CITNP.evaluation.tasks` (`make_context`); the names are re-exported here."""

import warnings

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
from CITNP.evaluation.tasks import make_context  # noqa: F401

warnings.warn(
    "CITNP.evaluation.oracle_lingauss is deprecated; use CITNP.evaluation.oracle "
    "and CITNP.evaluation.tasks",
    DeprecationWarning,
    stacklevel=2,
)
