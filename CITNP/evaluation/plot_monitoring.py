"""Deprecated: use `python -m CITNP.evaluation plot-monitoring`."""

import warnings

from CITNP.evaluation.__main__ import plot_monitoring_cli as main
from CITNP.evaluation.plots import plot_monitoring

__all__ = ["main", "plot_monitoring"]

warnings.warn(
    "CITNP.evaluation.plot_monitoring is deprecated; "
    "use `python -m CITNP.evaluation plot-monitoring`",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
