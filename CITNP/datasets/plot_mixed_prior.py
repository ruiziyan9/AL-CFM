"""Deprecated: moved to CITNP.prior_diagnostics.summary.

    python -m CITNP.prior_diagnostics summary --config <cfg> --out prior.png
"""

import warnings

from CITNP.prior_diagnostics.summary import collect, main, plot  # noqa: F401

if __name__ == "__main__":
    warnings.warn(
        "CITNP.datasets.plot_mixed_prior moved to "
        "`python -m CITNP.prior_diagnostics summary`",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
