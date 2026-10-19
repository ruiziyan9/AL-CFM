"""Deprecated: use `python -m CITNP.evaluation bank`."""

import warnings

from CITNP.evaluation.__main__ import bank_cli as main
from CITNP.evaluation.tasks import BUCKETS, build_bank, n_int_bucket  # noqa: F401

warnings.warn(
    "CITNP.evaluation.oracle_bank is deprecated; use `python -m CITNP.evaluation bank`",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
