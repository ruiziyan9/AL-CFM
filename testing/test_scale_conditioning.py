"""Scale conditioning (S4): the model receives the standardisation statistics,
so the standardised context plus (log std, mean / std) maps one-to-one back to
the raw context, as the exact oracle sees it."""

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from CITNP.evaluation.model_api import predict_mixture
from CITNP.experiments.mixed_train import MODEL_KEYS, apply_model_config
from CITNP.models.mixed_context_model import INTERVENTION_ENCODINGS
from CITNP.models.scale_conditioning import SCALE_PARAM_PREFIX

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_trainer import D, grouped_batch, prep, tiny_model_config

CONFIGS = Path(__file__).resolve().parents[1] / "CITNP" / "experiments" / "configs"
ENCODINGS = sorted(INTERVENTION_ENCODINGS)  # r1, layerwise, extra_node


def _models(encoding, seed=0):
    """A "none" model and a freshly built "stats" model with copied weights."""
    cls = INTERVENTION_ENCODINGS[encoding]
    cfg = tiny_model_config()
    torch.manual_seed(seed)
    none = cls(cfg).eval()
    torch.manual_seed(seed)
    stats = cls(dataclasses.replace(cfg, scale_conditioning="stats")).eval()
    res = stats.load_state_dict(none.state_dict(), strict=False)
    assert not res.unexpected_keys
    assert res.missing_keys and all(
        k.startswith(SCALE_PARAM_PREFIX) for k in res.missing_keys
    )
    return none, stats


def _forward(model, p, scale=None):
    return model(
        context_data=p["context"],
        target_data=p["target"],
        intervention_index=p["intervention_index"],
        outcome_index=p["outcome_index"],
        context_int_mask=p["context_int_mask"],
        context_scale=scale,
    )


def test_reconstruction_of_raw_context():
    batch = grouped_batch(G=2, B=6)
    out = prep(batch, normalise=True)
    B, G = 6, 2
    assert out["context_scale"].shape == (B * G, D, 2)
    scale = out["context_scale"].double()
    log_std, mean_over_std = scale[..., 0], scale[..., 1]  # [B*G, D]
    std = log_std.exp()
    ctx_std = out["context"][..., 0].double()  # [B*G, Nc, D]
    raw = ctx_std * std[:, None] + (mean_over_std * std)[:, None]
    want = batch["ctx_x"].double().repeat_interleave(G, dim=0)
    # both observational and interventional rows are checked; the statistics
    # are float32, so the match is to float32 rounding
    assert out["context_int_mask"].any() and not out["context_int_mask"].all()
    torch.testing.assert_close(raw, want, rtol=1e-5, atol=1e-5)
    # groups of one task share the statistics
    s = out["context_scale"].view(B, G, D, 2)
    assert torch.equal(s[:, 0], s[:, 1])


def test_no_normalisation_gives_zeros():
    out = prep(grouped_batch(G=2, B=4), normalise=False)
    assert torch.count_nonzero(out["context_scale"]) == 0


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_zero_init_matches_none_model(encoding):
    none, stats = _models(encoding)
    p = prep(grouped_batch(G=2, B=4), normalise=True)
    scale = torch.randn_like(p["context_scale"])  # arbitrary, deliberately not real
    with torch.no_grad():
        a, b = _forward(none, p), _forward(stats, p, scale)
    for x, y in zip((a.pred_mean, a.pred_std, a.weights), (b.pred_mean, b.pred_std, b.weights)):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-6)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_none_model_has_no_new_parameters(encoding):
    none, _ = _models(encoding)
    assert not any("scale_embedding" in k for k in none.state_dict())
    p = prep(grouped_batch(G=2, B=4), normalise=True)
    with torch.no_grad():  # context_scale is ignored under "none"
        a = _forward(none, p)
        b = _forward(none, p, torch.randn_like(p["context_scale"]))
    torch.testing.assert_close(a.pred_mean, b.pred_mean, rtol=0, atol=0)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_stats_model_requires_scale(encoding):
    _, stats = _models(encoding)
    p = prep(grouped_batch(G=2, B=2), normalise=True)
    with pytest.raises(ValueError, match="context_scale"):
        _forward(stats, p, None)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_scale_path_is_live_after_one_step(encoding):
    _, stats = _models(encoding)
    stats.train()
    p = prep(grouped_batch(G=2, B=4), normalise=True)
    s1 = p["context_scale"]
    s2 = s1 + torch.tensor([0.7, -0.4])
    opt = torch.optim.Adam(stats.parameters(), lr=1e-2)
    out = _forward(stats, p, s1)
    loss = stats.calculate_loss(
        model_output=out, target=p["target"], outcome_index=p["outcome_index"], test=False
    )
    loss.backward()
    opt.step()
    stats.eval()
    with torch.no_grad():
        a, b = _forward(stats, p, s1), _forward(stats, p, s2)
    assert (a.pred_mean - b.pred_mean).abs().max() > 1e-6


def test_extra_node_metadata_column_gets_no_scale():
    from CITNP.models.scale_conditioning import ScaleConditioning

    mod = ScaleConditioning(8, 2)
    torch.nn.init.normal_(mod.mlp[-1].weight)
    full = torch.zeros(2, 5, D + 1, 8)
    out = mod.add_to(full, torch.randn(2, D, 2))
    assert out[:, :, D].abs().max() == 0 and out[:, :, :D].abs().max() > 0


def test_predict_mixture_with_stats_model():
    _, stats = _models("extra_node")
    rng = np.random.default_rng(0)
    ctx_x = 5.0 * rng.normal(size=(30, D)) + 2.0
    ctx_int = np.zeros((30, D), dtype=bool)
    mu, sd, logw = predict_mixture(
        stats, ctx_x, ctx_int, 0, 1, np.linspace(-1, 1, 4), device="cpu"
    )
    assert mu.shape[:2] == (1, 4) and torch.isfinite(mu).all() and (sd > 0).all()


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*_stats.yaml")))
def test_stats_configs_are_valid(name):
    args = argparse.Namespace(**{k: None for k in MODEL_KEYS}, model_config=CONFIGS / name)
    apply_model_config(args)
    assert args.scale_conditioning == "stats"
