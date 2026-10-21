"""
Diagnostics of the mixed-regime prior DGP (no trained model needed).

Command line:  python -m CITNP.prior_diagnostics {examples,obs-vs-int,stats,summary} -h

Modules:
    stats       numpy-only statistics: DAG relations, edge strengths, effect gaps
    examples    one figure per task: weighted DAG + pairwise scatters
    obs_vs_int  one SCM: p(Y), p(Y | X~=x), p(Y | do(X=x))
    population  relation fractions and effect sizes over many tasks
    summary     aggregate histograms of the design (D, K, regime sizes, ...)
    obs_vs_do   exact Y|T vs Y|do(T) by question class, and context usefulness
    linear_gaussian  exact moments of linearnonidentifiable tasks
"""

from CITNP.prior_diagnostics.examples import plot_graph, plot_task, save_examples
from CITNP.prior_diagnostics.linear_gaussian import (
    LinearGaussianSCM,
    scm_from_debug,
)
from CITNP.prior_diagnostics.obs_vs_do import (
    collect_obs_vs_do,
    plot_obs_vs_do,
    select_example,
)
from CITNP.prior_diagnostics.obs_vs_int import plot_obs_vs_int, probe_task
from CITNP.prior_diagnostics.population import (
    collect_effects,
    collect_pairs,
    plot_stats,
)
from CITNP.prior_diagnostics.stats import (
    QUESTION_CLASSES,
    RELATIONS,
    context_role,
    edge_strengths,
    effect_gaps,
    path_lengths,
    query_pair_records,
    question_class,
    relation,
)

__all__ = [
    "QUESTION_CLASSES",
    "LinearGaussianSCM",
    "collect_obs_vs_do",
    "context_role",
    "plot_obs_vs_do",
    "question_class",
    "scm_from_debug",
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
    "select_example",
]
