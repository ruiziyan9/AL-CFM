"""reg_axis_full wiring: configs, mixed_train.build_trainer, resolved YAML and
model_api.load_model."""

from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.evaluation.model_api import load_model
from CITNP.experiments.mixed_train import build_trainer, parse_args
from CITNP.models.regime_axis import group_regimes
from CITNP.models.regime_axis_full import RegimeAxisFullMixedContextMoGModel

CONFIGS = (
    Path(__file__).resolve().parents[1] / "CITNP" / "experiments" / "configs"
)
SMOKE = CONFIGS / "smoke_d3_model_reg_axis_full.yaml"


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


@pytest.mark.parametrize("same_numregimes", [True, False])
def test_reg_axis_full_run_resolved_yaml_loads(tmp_path, same_numregimes):
    """Train two steps, then rebuild the model from the resolved YAML."""
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    extra = ["--same_numregimes_per_batch"] if same_numregimes else []
    trainer, _ = build_trainer(_args(tmp_path, SMOKE, *extra), cfg)
    assert type(trainer.model) is RegimeAxisFullMixedContextMoGModel
    if same_numregimes:
        for batch in trainer.train_dataset:
            assert group_regimes(batch["ctx_int"]).slot_present.all()
    trainer.train()
    assert trainer.global_step == 2
    run = tmp_path / "results" / "r"
    resolved = yaml.safe_load((run / "resolved_model_config.yaml").read_text())
    assert resolved["intervention_encoding"] == "reg_axis_full"
    assert resolved["model_class"] == "RegimeAxisFullMixedContextMoGModel"
    assert resolved["reg_axis_count_feature"] is True
    assert "reg_axis_num_memory_tokens" not in resolved
    assert "same_numregimes_per_batch" not in resolved  # a data option
    model = load_model(
        str(run / "last_model.pt"), str(run / "resolved_model_config.yaml"), "cpu"
    )
    assert type(model) is RegimeAxisFullMixedContextMoGModel
    saved = torch.load(run / "last_model.pt", map_location="cpu")
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved["model_state_dict"][k]), k


@pytest.mark.parametrize(
    "name", ["smoke_d3_model_reg_axis", "oracle_d3_model_m_reg_axis_stats"]
)
def test_config_differs_from_reg_axis_only_as_specified(name):
    """Same as the reg_axis config except for the encoding and the removed
    memory-token key."""
    reg_axis = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    full = yaml.safe_load(
        (
            CONFIGS / f"{name.replace('reg_axis', 'reg_axis_full')}.yaml"
        ).read_text()
    )
    assert reg_axis.pop("intervention_encoding") == "reg_axis"
    assert full.pop("intervention_encoding") == "reg_axis_full"
    assert "reg_axis_num_memory_tokens" not in full
    reg_axis.pop("reg_axis_num_memory_tokens")
    assert full == reg_axis


def test_init_from_upstream_rejected(tmp_path):
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    args = _args(tmp_path, SMOKE, "--init_from_upstream", "missing.pt")
    with pytest.raises(ValueError, match="init_from_upstream"):
        build_trainer(args, cfg)


def test_memory_token_key_rejected(tmp_path):
    yml = tmp_path / "m.yaml"
    m = yaml.safe_load(SMOKE.read_text())
    m["reg_axis_num_memory_tokens"] = 4
    yml.write_text(yaml.safe_dump(m))
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    with pytest.raises(ValueError, match="reg_axis_num_memory_tokens.*no memory"):
        build_trainer(_args(tmp_path, yml), cfg)


@pytest.mark.parametrize("encoding", ["r1", "layerwise", "extra_node"])
def test_reg_axis_keys_stay_rejected_elsewhere(tmp_path, encoding):
    yml = tmp_path / "m.yaml"
    m = yaml.safe_load(SMOKE.read_text())
    m["intervention_encoding"] = encoding
    yml.write_text(yaml.safe_dump(m))
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    with pytest.raises(ValueError, match="only apply to the reg_axis"):
        build_trainer(_args(tmp_path, yml), cfg)


def test_command_line_choice():
    args = parse_args(
        [
            "--data_config", "x.yaml",
            "--intervention_encoding", "reg_axis_full",
        ]
    )  # fmt: skip
    assert args.intervention_encoding == "reg_axis_full"
