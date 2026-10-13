"""Tests for CITNP/models/mixed_context_model.py (R1 Part B)."""

import numpy as np
import pytest
import torch

from CITNP.datasets.mixed_regime_generator import (
    MixedRegimeConfig,
    generate_task,
)
from CITNP.models.causalinferencemodel import MoGCausalInferenceModel
from CITNP.models.mixed_context_model import (
    MixedCausalInfEncoder,
    MixedContextMoGModel,
    load_upstream_checkpoint,
)
from CITNP.models.nncomponents import MoGPredictor
from CITNP.utils.configs import CausalInfModelConfig
from CITNP.utils.loss import MixtureGaussianLoss

D, NC, NQ, B = 3, 20, 6, 4


def tiny_config() -> CausalInfModelConfig:
    return CausalInfModelConfig(
        d_model=32,
        emb_depth=2,
        dim_feedforward=64,
        nhead=4,
        dropout=0.0,
        num_layers_encoder=4,
        num_nodes=D,
        sample_attn_mode="MHCA",
        linear_attention=False,
        mean_loss_across_samples=False,
        device="cpu",
        dtype="float32",
        num_mixture_components=3,
    )


def models():
    torch.manual_seed(0)
    up = MoGCausalInferenceModel(tiny_config()).eval()
    mixed = MixedContextMoGModel(tiny_config()).eval()
    load_upstream_checkpoint(mixed, up.state_dict())
    return up, mixed


def inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randn(B, NC, D, 1, generator=g)
    tgt = torch.randn(B, NQ, D, 1, generator=g)
    T = torch.tensor([0, 1, 2, 0])
    Y = torch.tensor([1, 2, 0, 2])
    masks = {
        "none": torch.zeros(B, NC, D, dtype=torch.bool),
        "random": torch.rand(B, NC, D, generator=g) < 0.3,
        "all": torch.ones(B, NC, D, dtype=torch.bool),
    }
    return ctx, tgt, T, Y, masks


def outputs(model, ctx, tgt, T, Y, mask=None):
    kw = {} if mask is None else {"context_int_mask": mask}
    out = model(
        context_data=ctx,
        target_data=tgt,
        intervention_index=T,
        outcome_index=Y,
        **kw,
    )
    return torch.cat([out.pred_mean, out.pred_std, out.weights], dim=-1)


def randomise_flag(mixed, seed=7):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mixed.encoder.context_intervened.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def test_mro_has_mog_head_and_mixed_encoder():
    _, mixed = models()
    assert isinstance(mixed.encoder, MixedCausalInfEncoder)
    assert isinstance(mixed.predictor, MoGPredictor)
    assert isinstance(mixed.loss, MixtureGaussianLoss)


@pytest.mark.parametrize("mask_name", ["none", "random", "all"])
def test_zero_init_equivalence(mask_name):
    up, mixed = models()
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        ref = outputs(up, ctx, tgt, T, Y)
        got = outputs(mixed, ctx, tgt, T, Y, masks[mask_name])
    assert torch.allclose(ref, got, atol=1e-6)


def test_flag_matters_when_trained():
    up, mixed = models()
    randomise_flag(mixed)
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        ref = outputs(up, ctx, tgt, T, Y)
        assert torch.allclose(
            ref, outputs(mixed, ctx, tgt, T, Y, masks["none"]), atol=1e-6
        )
        assert torch.allclose(
            ref, outputs(mixed, ctx, tgt, T, Y), atol=1e-6
        )  # None
        one = masks["none"].clone()
        one[:, 0, 0] = True
        assert not torch.allclose(
            ref, outputs(mixed, ctx, tgt, T, Y, one), atol=1e-4
        )


def test_context_order_invariance():
    _, mixed = models()
    randomise_flag(mixed)
    ctx, tgt, T, Y, masks = inputs()
    perm = torch.randperm(NC, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a = outputs(mixed, ctx, tgt, T, Y, masks["random"])
        b = outputs(mixed, ctx[:, perm], tgt, T, Y, masks["random"][:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_no_target_leakage():
    _, mixed = models()
    randomise_flag(mixed)
    ctx, tgt, T, Y, masks = inputs()
    tgt = tgt.clone().requires_grad_(True)
    out = outputs(mixed, ctx, tgt, T, Y, masks["random"])
    out.sum().backward()
    grad = tgt.grad[..., 0]  # [B, NQ, D]
    for b in range(B):
        non_treat = [d for d in range(D) if d != int(T[b])]
        assert (grad[b][:, non_treat] == 0).all()
    assert grad.abs().sum() > 0  # the treatment column does matter


def test_checkpoint_loading_keys(tmp_path):
    up, mixed = models()
    expected = {
        k
        for k in mixed.state_dict()
        if k.startswith("encoder.context_intervened.")
    }
    assert expected  # the new parameters exist
    # Raw state dict, CheckpointCallback-style dict, and a file path.
    for src in (up.state_dict(), {"model_state_dict": up.state_dict()}):
        res = load_upstream_checkpoint(MixedContextMoGModel(tiny_config()), src)
        assert set(res.missing_keys) == expected and not res.unexpected_keys
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": up.state_dict()}, path)
    res = load_upstream_checkpoint(MixedContextMoGModel(tiny_config()), path)
    assert set(res.missing_keys) == expected
    # Anything beyond the new parameters is an error.
    bad = dict(up.state_dict())
    bad["extra.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected"):
        load_upstream_checkpoint(MixedContextMoGModel(tiny_config()), bad)
    partial = {
        k: v for k, v in up.state_dict().items() if not k.startswith("predictor.")
    }
    with pytest.raises(RuntimeError, match="mismatch"):
        load_upstream_checkpoint(MixedContextMoGModel(tiny_config()), partial)


def _real_batch():
    """One fixed batch of grouped G=1 linear tasks, as model inputs."""
    cfg = MixedRegimeConfig(
        "linearfixedstd",
        [D],
        ["ER"],
        {D: [2, 3]},
        n_ctx_range=(64, 64),
        n_qry=16,
        qry_mode="grouped",
        qry_groups=[1],
    )
    tasks = [generate_task(cfg, s, n_groups=1) for s in range(8)]
    ctx = torch.tensor(np.stack([t.ctx_x for t in tasks]))[..., None]
    cint = torch.tensor(np.stack([t.ctx_int for t in tasks]))
    tgt = torch.tensor(np.stack([t.qry_x for t in tasks]))[..., None]
    T = torch.tensor([t.meta["group_treatments"][0] for t in tasks])
    Y = (T + 1) % D
    return ctx, cint, tgt, T, Y


def test_overfits_one_batch():
    torch.manual_seed(0)
    model = MixedContextMoGModel(tiny_config()).train()
    ctx, cint, tgt, T, Y = _real_batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn():
        out = model(ctx, tgt, T, Y, context_int_mask=cint)
        return model.calculate_loss(out, target=tgt, outcome_index=Y)

    first = loss_fn().item()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    last = loss_fn().item()
    assert np.isfinite(last)
    assert first - last > 0.5 * abs(first), (first, last)
