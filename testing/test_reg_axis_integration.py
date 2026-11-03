"""reg_axis wiring: configs, mixed_train.build_trainer, resolved YAML and
model_api.load_model."""

from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.evaluation.model_api import load_model
from CITNP.experiments.mixed_train import build_trainer, parse_args
from CITNP.models.regime_axis import RegimeAxisMixedContextMoGModel

CONFIGS = (
    Path(__file__).resolve().parents[1] / "CITNP" / "experiments" / "configs"
)
SMOKE = CONFIGS / "smoke_d3_model_reg_axis.yaml"


def _args(tmp_path, model_config, *extra):
    return parse_args(
        [
            "--data_config", str(CONFIGS / "smoke_d3.yaml"),
            "--model_config", str(model_config),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", "4", "--iterations_per_epoch", "2",
            "--epochs", "1", "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--save_checkpoint_every_n_steps", "1000",
            "--results_path", str(tmp_path), "--run_name", "r", *extra,
        ]
    )  # fmt: skip


def test_reg_axis_run_resolved_yaml_loads(tmp_path):
    """Train two steps, then rebuild the model from the resolved YAML."""
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    trainer, _ = build_trainer(_args(tmp_path, SMOKE), cfg)
    assert type(trainer.model) is RegimeAxisMixedContextMoGModel
    trainer.train()
    run = tmp_path / "results" / "r"
    resolved = yaml.safe_load((run / "resolved_model_config.yaml").read_text())
    assert resolved["intervention_encoding"] == "reg_axis"
    assert resolved["reg_axis_num_memory_tokens"] == 4
    assert resolved["reg_axis_count_feature"] is True
    model = load_model(
        str(run / "last_model.pt"), str(run / "resolved_model_config.yaml"), "cpu"
    )
    saved = torch.load(run / "last_model.pt", map_location="cpu")
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved["model_state_dict"][k]), k


def test_oracle_config_differs_only_in_encoding():
    lw = yaml.safe_load(
        (CONFIGS / "oracle_d3_model_m_layerwise_stats.yaml").read_text()
    )
    ra = yaml.safe_load(
        (CONFIGS / "oracle_d3_model_m_reg_axis_stats.yaml").read_text()
    )
    lw.pop("model_class")
    assert ra.pop("intervention_encoding") == "reg_axis"
    ra = {k: v for k, v in ra.items() if not k.startswith("reg_axis_")}
    assert ra == lw


def test_init_from_upstream_rejected(tmp_path):
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    args = _args(tmp_path, SMOKE, "--init_from_upstream", "missing.pt")
    with pytest.raises(ValueError, match="init_from_upstream"):
        build_trainer(args, cfg)


def test_reg_axis_keys_need_reg_axis(tmp_path):
    yml = tmp_path / "m.yaml"
    m = yaml.safe_load(SMOKE.read_text())
    m["intervention_encoding"] = "layerwise"
    yml.write_text(yaml.safe_dump(m))
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    with pytest.raises(ValueError, match="only apply to the reg_axis"):
        build_trainer(_args(tmp_path, yml), cfg)
