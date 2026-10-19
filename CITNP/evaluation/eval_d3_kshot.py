"""Deprecated: use `python -m CITNP.evaluation kshot` (and `plot-kshot`)."""

import warnings

from CITNP.evaluation.__main__ import kshot_cli as main
from CITNP.evaluation.metrics import mix_logpdf, obs_cov
from CITNP.evaluation.tasks import (
    DESIGNS,
    FIXED_NODE,
    MAIN_QUERY,
    QUERIES,
    ROLE,
    confounder_scm,
    identified_queries,
    make_pools,
    qname,
    query_type,
    random_confounder_scm,
    standardise,
)

__all__ = [
    "DESIGNS",
    "FIXED_NODE",
    "MAIN_QUERY",
    "QUERIES",
    "ROLE",
    "confounder_scm",
    "identified_queries",
    "main",
    "make_pools",
    "mix_logpdf",
    "obs_cov",
    "qname",
    "query_type",
    "random_confounder_scm",
    "standardise",
]

warnings.warn(
    "CITNP.evaluation.eval_d3_kshot is deprecated; "
    "use `python -m CITNP.evaluation kshot`",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
