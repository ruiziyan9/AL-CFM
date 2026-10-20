"""
Diagnostics of the mixed-regime prior DGP (no trained model needed).

Command line:  python -m CITNP.prior_diagnostics {examples,obs-vs-int,stats,summary} -h

Modules:
    stats       numpy-only statistics: DAG relations, edge strengths, effect gaps
    examples    one figure per task: weighted DAG + pairwise scatters
    obs_vs_int  one SCM: p(Y), p(Y | X~=x), p(Y | do(X=x))
    population  relation fractions and effect sizes over many tasks
    summary     aggregate histograms of the design (D, K, regime sizes, ...)
"""

from CITNP.prior_diagnostics.examples import plot_graph, plot_task, save_examples
from CITNP.prior_diagnostics.obs_vs_int import plot_obs_vs_int, probe_task
from CITNP.prior_diagnostics.population import (
    collect_effects,
    collect_pairs,
    plot_stats,
)
from CITNP.prior_diagnostics.stats import (
    RELATIONS,
    edge_strengths,
    effect_gaps,
    path_lengths,
    query_pair_records,
    relation,
)

__all__ = [
    "RELATIONS",
    "collect_effects",
    "collect_pairs",
    "edge_strengths",
    "effect_gaps",
    "path_lengths",
    "plot_graph",
    "plot_obs_vs_int",
    "plot_stats",
    "plot_task",
    "probe_task",
    "query_pair_records",
    "relation",
    "save_examples",
]
