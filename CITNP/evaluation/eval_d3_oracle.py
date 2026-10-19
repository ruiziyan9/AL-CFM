"""Deprecated: use `python -m CITNP.evaluation scm-suite`."""

import warnings

from CITNP.evaluation.__main__ import scm_suite_cli as main
from CITNP.evaluation.model_api import load_model  # noqa: F401  (re-export)
from CITNP.evaluation.tasks import context_designs as designs  # noqa: F401
from CITNP.evaluation.tasks import empirical_graph_prior  # noqa: F401
from CITNP.evaluation.tasks import scm_suite as scms  # noqa: F401

warnings.warn(
    "CITNP.evaluation.eval_d3_oracle is deprecated; "
    "use `python -m CITNP.evaluation scm-suite`",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
