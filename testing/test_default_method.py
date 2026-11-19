"""The default method (layer-wise encoding with the shared flag network) and
the model config every shipped model YAML builds.

The table below is hard-coded on purpose: a change of default, or a new or
edited model config, must show up here as a deliberate edit.
"""

import argparse
import sys
from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.evaluation.model_api import build_model, load_model
from CITNP.experiments.mixed_train import (
    MODEL_KEYS,
    _share_intervention_encoder,
    apply_model_config,
)
from CITNP.models.mixed_context_model import (
    LayerwiseMixedContextMoGModel,
    resolve_model_class,
)
from CITNP.utils.config_files import load_config_mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import tiny_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "CITNP" / "experiments" / "configs"
FIXTURES = REPO / "testing" / "fixtures"

# path -> (model class name, share_intervention_encoder of the built encoder;
# None where the key does not apply, i.e. every encoding but layer-wise)
EXPECTED = {
    CONFIGS / "smoke_d3_model.yaml": ("LayerwiseMixedContextMoGModel", True),
    CONFIGS / "smoke_d3_model_extra_node.yaml": (
        "ExtraNodeMixedContextMoGModel", None,
    ),
    CONFIGS / "smoke_d3_model_extra_node_stats.yaml": (
        "ExtraNodeMixedContextMoGModel", None,
    ),
    CONFIGS / "smoke_d3_model_reg_axis.yaml": (
        "RegimeAxisMixedContextMoGModel", None,
    ),
    CONFIGS / "smoke_d3_model_reg_axis_full.yaml": (
        "RegimeAxisFullMixedContextMoGModel", None,
    ),
    CONFIGS / "oracle_d3_model_m_extra_node.yaml": (
        "ExtraNodeMixedContextMoGModel", None,
    ),
    CONFIGS / "oracle_d3_model_m_extra_node_stats.yaml": (
        "ExtraNodeMixedContextMoGModel", None,
    ),
    CONFIGS / "oracle_d3_model_m_layerwise.yaml": (
        "LayerwiseMixedContextMoGModel", False,
    ),
    CONFIGS / "oracle_d3_model_m_layerwise_stats.yaml": (
        "LayerwiseMixedContextMoGModel", False,
    ),
    CONFIGS / "oracle_d3_model_m_layerwise_stats_shared.yaml": (
        "LayerwiseMixedContextMoGModel", True,
    ),
    CONFIGS / "oracle_d3_model_m_reg_axis_stats.yaml": (
        "RegimeAxisMixedContextMoGModel", None,
    ),
    CONFIGS / "oracle_d3_model_m_reg_axis_full_stats.yaml": (
        "RegimeAxisFullMixedContextMoGModel", None,
    ),
    # fixtures of the abandoned r1 encoding (evaluation regression tests)
    FIXTURES / "r1_model_s.yaml": ("MixedContextMoGModel", None),
    FIXTURES / "r1_model_m.yaml": ("MixedContextMoGModel", None),
}  # fmt: skip


def _is_model_config(path: Path) -> bool:
    return set(load_config_mapping(path)) <= set(MODEL_KEYS)


def _args(model_config: Path) -> argparse.Namespace:
    base = {k: None for k in MODEL_KEYS}
    base["model_config"] = str(model_config)
    return argparse.Namespace(**base)


def test_table_lists_every_shipped_model_config():
    shipped = {p for p in CONFIGS.glob("*.yaml") if _is_model_config(p)}
    assert shipped == {p for p in EXPECTED if p.parent == CONFIGS}


def test_no_experiment_config_uses_r1():
    for path in EXPECTED:
        if path.parent == CONFIGS:
            assert EXPECTED[path][0] != "MixedContextMoGModel", path.name


@pytest.mark.parametrize(
    "path", sorted(EXPECTED), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_config_builds_expected_model(path):
    class_name, share = EXPECTED[path]
    model = build_model(str(path), "cpu", num_nodes=3)
    assert type(model).__name__ == class_name
    assert getattr(model.encoder, "share_intervention_encoder", None) is share
    if share is not None:
        assert hasattr(model.encoder, "flag_code") is not share
    # the training script resolves the same class and the same option
    args = _args(path)
    apply_model_config(args)
    cls, _ = resolve_model_class(args.intervention_encoding, args.model_class)
    assert cls.__name__ == class_name
    if share is not None:
        assert _share_intervention_encoder(args) is share


def _encoding_free_yaml(tmp_path) -> Path:
    c = tiny_config()
    settings = {
        "d_model": c.d_model,
        "emb_depth": c.emb_depth,
        "dim_feedforward": c.dim_feedforward,
        "nhead": c.nhead,
        "num_layers_encoder": c.num_layers_encoder,
        "sample_attn_mode": c.sample_attn_mode,
        "num_mixture_components": c.num_mixture_components,
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def test_yaml_without_encoding_builds_shared_layerwise(tmp_path):
    path = _encoding_free_yaml(tmp_path)
    model = build_model(str(path), "cpu", num_nodes=3)
    assert type(model) is LayerwiseMixedContextMoGModel
    assert model.encoder.share_intervention_encoder is True
    assert not hasattr(model.encoder, "flag_code")
    args = _args(path)
    apply_model_config(args)
    cls, name = resolve_model_class(args.intervention_encoding, args.model_class)
    assert cls is LayerwiseMixedContextMoGModel and name == "layerwise"
    assert _share_intervention_encoder(args) is True


def test_old_separate_layerwise_run_without_key_raises_clearly(tmp_path):
    """A layer-wise run saved before share_intervention_encoder existed (its
    resolved config has no entry) must not be rebuilt as shared silently."""
    cfg = tiny_config()
    cfg.share_intervention_encoder = False
    torch.manual_seed(0)
    old = LayerwiseMixedContextMoGModel(cfg)
    ckpt = tmp_path / "last_model.pt"
    torch.save({"model_state_dict": old.state_dict()}, ckpt)
    path = _encoding_free_yaml(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["model_class"] = "LayerwiseMixedContextMoGModel"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="share_intervention_encoder: false"):
        load_model(str(ckpt), str(path), "cpu", num_nodes=3)
    settings["share_intervention_encoder"] = False
    path.write_text(yaml.safe_dump(settings))
    loaded = load_model(str(ckpt), str(path), "cpu", num_nodes=3)
    assert loaded.encoder.share_intervention_encoder is False
