"""Files mixed_train.build_trainer writes to the run directory:
resolved_model_config.yaml (model keys only, reusable as --model_config) and
resolved_data_options.yaml (run-level data options)."""

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.experiments.mixed_train import (
    MODEL_KEYS,
    RESOLVED_DATA_OPTIONS_FILE,
    build_trainer,
    parse_args,
)

CONFIGS = (
    Path(__file__).resolve().parents[1] / "CITNP" / "experiments" / "configs"
)
MODEL_CONFIGS = [
    "smoke_d3_model.yaml",  # r1
    "smoke_d3_model_extra_node_stats.yaml",
    "oracle_d3_model_m_layerwise_stats.yaml",  # legacy model_class key
    "oracle_d3_model_m_layerwise_stats_shared.yaml",
    "smoke_d3_model_reg_axis.yaml",
    "smoke_d3_model_reg_axis_full.yaml",
]


def _trainer(tmp_path, run_name, model_config, *extra):
    args = parse_args(
        [
            "--data_config", str(CONFIGS / "smoke_d3.yaml"),
            "--model_config", str(model_config),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", "4", "--iterations_per_epoch", "1",
            "--epochs", "1", "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--results_path", str(tmp_path), "--run_name", run_name, *extra,
        ]
    )  # fmt: skip
    trainer, _ = build_trainer(args, load_config(CONFIGS / "smoke_d3.yaml"))
    return trainer, tmp_path / "results" / run_name


@pytest.mark.parametrize("model_config", MODEL_CONFIGS)
@pytest.mark.parametrize("flag", [(), ("--same_numregimes_per_batch",)])
def test_resolved_model_config_round_trip(tmp_path, model_config, flag):
    """resolved_model_config.yaml, passed back unchanged as --model_config,
    builds the same model and resolves to the same file."""
    first, first_run = _trainer(tmp_path, "first", CONFIGS / model_config, *flag)
    resolved_path = first_run / "resolved_model_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    assert set(resolved) <= set(MODEL_KEYS)  # nothing apply_model_config rejects
    second, second_run = _trainer(tmp_path, "second", resolved_path)
    assert type(second.model) is type(first.model)
    a, b = first.model.state_dict(), second.model.state_dict()
    assert list(a) == list(b)
    assert all(a[k].shape == b[k].shape for k in a)
    again = yaml.safe_load(
        (second_run / "resolved_model_config.yaml").read_text()
    )
    assert again == resolved


@pytest.mark.parametrize("flag", [False, True])
def test_resolved_data_options(tmp_path, flag):
    extra = ["--same_numregimes_per_batch"] if flag else []
    _, run = _trainer(tmp_path, "r", CONFIGS / "smoke_d3_model.yaml", *extra)
    options = yaml.safe_load((run / RESOLVED_DATA_OPTIONS_FILE).read_text())
    assert options == {"same_numregimes_per_batch": flag}
    resolved = yaml.safe_load((run / "resolved_model_config.yaml").read_text())
    assert "same_numregimes_per_batch" not in resolved
