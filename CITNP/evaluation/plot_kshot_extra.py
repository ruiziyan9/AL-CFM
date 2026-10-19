"""
Extra diagnostic figures from the CSVs written by CITNP.evaluation.eval_d3_kshot.

    python -m CITNP.evaluation.plot_kshot_extra kshot_m_best_k50.csv

Writes <name>_flag.png, _varratio.png, _qtype.png and _by_t.png next to the
CSV; the plotting code lives in CITNP.evaluation.plots.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from CITNP.evaluation.plots import (  # noqa: E402, F401  (re-exports)
    plot_by_t,
    plot_flag,
    plot_qtype,
    plot_varratio,
)


def main():
    path = sys.argv[1]
    prefix = path.removesuffix(".csv")
    df = pd.read_csv(path)
    ks = sorted(df.k.unique())
    plot_flag(df, prefix, ks)
    plot_varratio(df, prefix, ks)
    plot_qtype(df, prefix, ks)
    try:
        bt = pd.read_csv(prefix + "_by_t.csv")
        plot_by_t(bt, prefix, ks)
    except FileNotFoundError:
        print(f"no {prefix}_by_t.csv, skipping the |t| figure")
    print(f"wrote {prefix}_flag.png, _varratio.png, _qtype.png, _by_t.png")


if __name__ == "__main__":
    main()
