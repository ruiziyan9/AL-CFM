"""`TrainingSupport` of a CITNP run, from its data config (and model).

From a `MixedRegimeConfig` (see `sample_design` in
CITNP/datasets/mixed_regime_generator.py):
- trained D: `num_variables`;
- context rows: `n_ctx_range`;
- an observational regime with at least n_obs_min + 1 rows when n_obs_min > 0
  (one row plus the reserve);
- at most min(k_max, max D) interventional regimes, each on a distinct node,
  every regime at least one row;
- do-values on the raw scale, the same distribution for every node:
  N(0, s^2), s ~ U(do_scale_range), or with probability p_wide_do
  Uniform(-w, w), w ~ U(wide_do_range); the support check uses the largest
  uniform range [-max w, max w] (for oracle_d5_iw_a15_sc.yaml, [-6, 6]), or
  +- 3 max s if wide do-values are switched off.
From the model: the largest D of an architecture with fixed variable-index
embeddings (the extra-node encoding), else unbounded.
"""

import torch

from causal_active_learning.support_check import TrainingSupport


def do_value_range(data_config) -> tuple[float, float]:
    """The do-value range the support check uses (module docstring)."""
    if float(getattr(data_config, "p_wide_do", 0.0)) > 0:
        w = float(data_config.wide_do_range[1])
    else:
        w = 3.0 * float(data_config.do_scale_range[1])
    return -w, w


def maximum_num_variables(model: torch.nn.Module | None) -> int | None:
    """The size of the model's variable-index embedding, if it has one
    (`extra_node` encoders: `index_embedding`), else None (unbounded)."""
    if model is None:
        return None
    for module in model.modules():
        embedding = getattr(module, "index_embedding", None)
        if isinstance(embedding, torch.nn.Embedding):
            return int(embedding.num_embeddings)
    return None


def training_support_from_data_config(
    data_config, model: torch.nn.Module | None = None
) -> TrainingSupport:
    """The training support of a run with this data config and model."""
    trained = tuple(int(d) for d in data_config.num_variables)
    n_obs_min = int(getattr(data_config, "n_obs_min", 0))
    return TrainingSupport(
        trained_num_variables=trained,
        maximum_num_variables=maximum_num_variables(model),
        context_size_range=(
            int(data_config.n_ctx_range[0]),
            int(data_config.n_ctx_range[1]),
        ),
        minimum_observational_rows=n_obs_min + 1 if n_obs_min > 0 else 0,
        maximum_interventional_regimes=min(int(data_config.k_max), max(trained)),
        rows_per_regime_range=(1, None),
        do_value_range=do_value_range(data_config),
        observational_queries_trained=float(
            getattr(data_config, "observational_query_probability", 0.0)
        )
        > 0,
    )
