"""`TrainingSupport` of a CITNP run, from its data config."""

from causal_active_learning.support_check import TrainingSupport


def training_support_from_data_config(
    data_config, num_variables: int
) -> TrainingSupport:
    """From a `MixedRegimeConfig` for D = num_variables (see
    `sample_design` in CITNP/datasets/mixed_regime_generator.py): context
    rows `n_ctx_range`; an observational regime with at least n_obs_min + 1
    rows when n_obs_min > 0; at most min(k_max, D) interventional regimes."""
    n_obs_min = int(getattr(data_config, "n_obs_min", 0))
    return TrainingSupport(
        context_size_range=(
            int(data_config.n_ctx_range[0]),
            int(data_config.n_ctx_range[1]),
        ),
        minimum_observational_rows=n_obs_min + 1 if n_obs_min > 0 else 0,
        maximum_interventional_regimes=min(
            int(data_config.k_max), int(num_variables)
        ),
    )
