"""Deprecated: use `python -m CITNP.evaluation bank`."""

import warnings

from CITNP.evaluation.__main__ import bank_cli as main
from CITNP.evaluation.tasks import BUCKETS, build_bank, n_int_bucket

__all__ = ["BUCKETS", "build_bank", "main", "n_int_bucket"]

warnings.warn(
    "CITNP.evaluation.oracle_bank is deprecated; "
    "use `python -m CITNP.evaluation bank`",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
