"""Tests for CITNP/evaluation/embedding_geometry.py on fresh, randomly
initialised extra-node encoders (structure only, nothing about learned
behaviour)."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

from CITNP.evaluation.embedding_geometry import (
    first_block_input_normalisation,
    metadata_embedding_geometry,
)
from CITNP.models.mixed_context_model import ExtraNodeMixedContextMoGModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import D, tiny_config  # noqa: E402

TARGET_SET_KEYS = {
    "target_sets",
    "raw_token_norms",
    "summed_index_embedding_norms",
    "projection_on_summed_index_embedding",
    "normalised_cosine_to_observational",
    "normalised_distance_to_observational",
    "min_pairwise_normalised_distance",
}


def extra_node_model(num_nodes: int = D, seed: int = 0):
    cfg = tiny_config()
    cfg.num_nodes = num_nodes
    torch.manual_seed(seed)
    return ExtraNodeMixedContextMoGModel(cfg).eval()


def test_keys_and_shapes():
    model = extra_node_model()
    result = metadata_embedding_geometry(model.encoder)
    assert result["num_variables"] == D
    assert result["d_model"] == model.d_model
    assert "LayerNorm" in result["normalisation"]
    assert len(result["index_embedding_norms"]) == D
    assert len(result["cosine_int_meta_vs_index"]) == D
    num_pairs = D * (D - 1) // 2
    for key, num_sets in (("single_target", D), ("two_targets", num_pairs)):
        stats = result[key]
        assert set(stats) == TARGET_SET_KEYS
        assert len(stats["target_sets"]) == num_sets
        for name in TARGET_SET_KEYS - {
            "target_sets",
            "min_pairwise_normalised_distance",
        }:
            assert len(stats[name]) == num_sets, (key, name)
        assert isinstance(stats["min_pairwise_normalised_distance"], float)
    assert result["single_target"]["target_sets"] == [[j] for j in range(D)]
    json.dumps(result)  # plain Python types only


def test_intervened_tokens_differ_from_observational():
    result = metadata_embedding_geometry(extra_node_model().encoder)
    for key in ("single_target", "two_targets"):
        assert all(
            d > 0 for d in result[key]["normalised_distance_to_observational"]
        )
        cosines = result[key]["normalised_cosine_to_observational"]
        assert all(c < 1 for c in cosines)


def test_single_target_token_projects_positively_on_its_index_embedding():
    result = metadata_embedding_geometry(extra_node_model().encoder)
    projections = result["single_target"]["projection_on_summed_index_embedding"]
    assert all(p > 0 for p in projections)


def test_raw_quantities_match_the_parameters():
    encoder = extra_node_model().encoder
    result = metadata_embedding_geometry(encoder)
    e_int = encoder.int_meta_embedding.detach().double()
    e = encoder.index_embedding.weight.detach().double()
    assert result["int_meta_embedding_norm"] == pytest.approx(float(e_int.norm()))
    assert result["single_target"]["raw_token_norms"][1] == pytest.approx(
        float((e_int + e[1]).norm())
    )
    assert result["two_targets"]["raw_token_norms"][0] == pytest.approx(
        float((e_int + e[0] + e[1]).norm())
    )


def test_normalisation_is_the_first_block_layer_norm():
    encoder = extra_node_model().encoder
    _, normalise = first_block_input_normalisation(encoder)
    token = torch.randn(encoder.int_meta_embedding.shape[0], dtype=torch.float64)
    norm = copy.deepcopy(encoder.encoder.sample_layers[0].norm1).double()
    expected = norm(token)
    torch.testing.assert_close(normalise(token), expected)


def test_post_norm_uses_raw_tokens():
    encoder = extra_node_model().encoder
    encoder.encoder.sample_layers[0].norm_first = False
    name, normalise = first_block_input_normalisation(encoder)
    assert "post-norm" in name
    token = torch.randn(4, dtype=torch.float64)
    assert torch.equal(normalise(token), token)


def test_fewer_variables_than_the_model_maximum():
    result = metadata_embedding_geometry(
        extra_node_model(num_nodes=4).encoder, num_variables=2
    )
    assert result["num_variables"] == 2
    assert len(result["single_target"]["target_sets"]) == 2
    assert result["two_targets"]["target_sets"] == [[0, 1]]
    assert result["two_targets"]["min_pairwise_normalised_distance"] is None


def test_other_encoders_are_rejected():
    from CITNP.models.mixed_context_model import MixedContextMoGModel

    with pytest.raises(TypeError, match="ExtraNode"):
        metadata_embedding_geometry(MixedContextMoGModel(tiny_config()).encoder)


def test_embedding_geometry_cli(tmp_path):
    import yaml  # type: ignore[import-untyped]

    from CITNP.evaluation.__main__ import main

    model = extra_node_model()
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    cfg = tiny_config()
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump({
            "d_model": cfg.d_model, "nhead": cfg.nhead,
            "num_layers_encoder": cfg.num_layers_encoder,
            "sample_attn_mode": cfg.sample_attn_mode,
            "dim_feedforward": cfg.dim_feedforward,
            "emb_depth": cfg.emb_depth,
            "num_mixture_components": cfg.num_mixture_components,
            "intervention_encoding": "extra_node",
        })  # fmt: skip
    )
    out = tmp_path / "geometry.json"
    main(["embedding-geometry", "--ckpt", str(ckpt), "--model_config",
          str(model_config), "--device", "cpu", "--out", str(out)])  # fmt: skip
    saved = json.loads(out.read_text())
    assert saved["geometry"] == metadata_embedding_geometry(model.encoder)
    assert saved["ckpt"] == str(ckpt)
