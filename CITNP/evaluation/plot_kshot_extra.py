"""Deprecated: use `python -m CITNP.evaluation plot-kshot` (all k-shot figures)."""

import warnings

from CITNP.evaluation.__main__ import plot_kshot_cli as main
from CITNP.evaluation.plots import plot_by_t, plot_flag, plot_qtype, plot_varratio

__all__ = ["main", "plot_by_t", "plot_flag", "plot_qtype", "plot_varratio"]

warnings.warn(
    "CITNP.evaluation.plot_kshot_extra is deprecated; "
    "use `python -m CITNP.evaluation plot-kshot`",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
