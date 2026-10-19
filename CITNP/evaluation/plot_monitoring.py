"""
Plot oracle_gap.csv and train_stats.csv for one or more runs.

    python -m CITNP.evaluation.plot_monitoring \
        /content/drive/MyDrive/alcfm/results/oracle_d3_s_colab \
        /content/drive/MyDrive/alcfm/results/oracle_d3_m_colab \
        --out monitoring.png

The plotting code lives in CITNP.evaluation.plots.plot_monitoring.
"""

import argparse

from CITNP.evaluation.plots import plot_monitoring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="monitoring.png")
    a = ap.parse_args()
    plot_monitoring(a.run_dirs, a.out)


if __name__ == "__main__":
    main()
