"""Tests for the reg_axis model (CITNP/models/regime_axis)."""

import sys
from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.evaluation.model_api import load_model
from CITNP.models.mixed_context_model import (
    INTERVENTION_ENCODINGS,
    LayerwiseMixedContextMoGModel,
    resolve_model_class,
)
from CITNP.models.regime_axis import (
    RegimeAxisEncoder,
    RegimeAxisMixedContextMoGModel,
    group_regimes,
)
from CITNP.models.regime_axis.pooling import pooling_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import tiny_config  # noqa: E402
from test_reg_axis_grouping import build_mask  # noqa: E402

D, NQ = 3, 5
# Tasks {obs}, {obs, do(X1)}, {obs, do(X1), do(X3)}, {do(X2)} (1-indexed
# variables): R_b = 1, 2, 3, 1, unequal regime sizes, Nc = 12.
MISSING = [
    [((), 12)],
    [((), 7), ((0,), 5)],
    [((), 4), ((0,), 5), ((2,), 3)],
    [((1,), 12)],
]
TOL = dict(atol=1e-10, rtol=0)


def model(seed=0, randomise=True, **cfg_over):
    cfg = tiny_config()
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    torch.manual_seed(seed)
    m = RegimeAxisMixedContextMoGModel(cfg).double().eval()
    if randomise:  # zero-initialised flag / scale layers get non-zero weights
        g = torch.Generator().manual_seed(seed + 100)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name.startswith(("encoder.flag.", "encoder.scale_embedding.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return m


def batch(specs=MISSING, seed=1, scale=False):
    mask, lab = build_mask(specs, D=D, seed=seed)
    B, Nc, _ = mask.shape
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randn(B, Nc, D, 1, generator=g, dtype=torch.float64)
    inp = dict(
        context_data=ctx,
        target_data=torch.randn(B, NQ, D, 1, generator=g, dtype=torch.float64),
        intervention_index=torch.arange(B) % D,
        outcome_index=(torch.arange(B) + 1) % D,
        context_int_mask=mask,
    )
    if scale:
        inp["context_scale"] = torch.randn(
            B, D, 2, generator=g, dtype=torch.float64
        )
    return inp, lab


def predict(m, **inp):
    with torch.no_grad():
        out = m(**inp)
    return torch.cat([out.pred_mean, out.pred_std, out.weights], dim=-1)


def hook(module, store, key):
    return module.register_forward_hook(
        lambda mod, args, out: store.__setitem__(key, out)
    )


# --------------------------------------------------------------------------- #
# Structure and registry
# --------------------------------------------------------------------------- #


def test_registry_and_structure():
    cls, name = resolve_model_class("reg_axis")
    assert cls is RegimeAxisMixedContextMoGModel and name == "reg_axis"
    assert INTERVENTION_ENCODINGS["reg_axis"] is cls
    m = model()
    assert isinstance(m.encoder, RegimeAxisEncoder)
    assert not hasattr(m.encoder, "encoder")  # upstream stack removed
    # default depth = num_layers_encoder // 2; last block has no feedback
    assert len(m.encoder.blocks) == tiny_config().num_layers_encoder // 2
    assert m.encoder.blocks[0].feedback is not None
    assert m.encoder.blocks[-1].feedback is None
    # role MLPs keep the upstream parameter names
    for role in ("context", "target"):
        for kind in ("variable", "intervention", "outcome"):
            assert f"encoder.{role}_{kind}.0.weight" in m.state_dict()


def test_rejected_options():
    with pytest.raises(ValueError, match="linear_attention"):
        model(linear_attention=True)
    with pytest.raises(ValueError, match="reg_axis_target_flag"):
        model(reg_axis_target_flag="nope")
    with pytest.raises(ValueError, match="at least one block"):
        model(reg_axis_num_blocks=0)
    m = model()
    inp, _ = batch()
    with pytest.raises(ValueError, match="variable_mask"):
        m(**inp, variable_mask=torch.ones(4, D))


def test_sample_attn_mode_ignored():
    """Same parameters and same function for either mode (only the random
    init differs, through the discarded upstream stack built by the base
    class)."""
    inp, _ = batch()
    a = model(sample_attn_mode="MHCA")
    b = model(sample_attn_mode="MHSA", seed=1)
    b.load_state_dict(a.state_dict())  # strict
    assert torch.allclose(predict(a, **inp), predict(b, **inp), **TOL)


# --------------------------------------------------------------------------- #
# Masked computation equals a per-regime loop
# --------------------------------------------------------------------------- #


def test_masked_equals_loop():
    m = model()
    block = m.encoder.blocks[0]
    inp, _ = batch()
    mask = inp["context_int_mask"]
    B, Nc, _ = mask.shape
    grouping = group_regimes(mask)
    g = torch.Generator().manual_seed(3)
    hc = torch.randn(B, Nc, D, 32, generator=g, dtype=torch.float64)
    with torch.no_grad():
        h2 = block.sample_attention(hc, grouping)
        mem = block.memories(h2, grouping)  # [B, D, R, K, d]
        for b in range(B):
            for r in range(grouping.num_slots):
                idx = (grouping.slot[b] == r).nonzero().flatten()
                if len(idx) == 0:
                    assert (mem[b, :, r] == 0).all()
                    continue
                sub = group_regimes(mask[b : b + 1, idx])
                ref2 = block.sample_attention(hc[b : b + 1, idx], sub)
                assert torch.allclose(h2[b, idx], ref2[0], atol=1e-6)
                ref_mem = block.memories(h2[b : b + 1, idx], sub)
                assert torch.allclose(mem[b, :, r], ref_mem[0, :, 0], atol=1e-6)


# --------------------------------------------------------------------------- #
# Symmetries
# --------------------------------------------------------------------------- #


def test_row_permutation():
    m = model()
    inp, _ = batch()
    ref = predict(m, **inp)
    perm = torch.randperm(inp["context_data"].shape[1])  # within and across
    shuffled = dict(inp)
    shuffled["context_data"] = inp["context_data"][:, perm]
    shuffled["context_int_mask"] = inp["context_int_mask"][:, perm]
    assert torch.allclose(predict(m, **shuffled), ref, **TOL)


def test_regime_relabel():
    m = model()
    inp, lab = batch()
    ref = predict(m, **inp)
    assert torch.allclose(predict(m, **inp, context_regime=lab), ref, **TOL)
    remap = torch.tensor([5, 2, 9])
    assert torch.allclose(
        predict(m, **inp, context_regime=remap[lab]), ref, **TOL
    )


def _append_row(inp, b_spec, cols):
    """Append one all-zero context row with target set `cols`."""
    out = dict(inp)
    row_x = torch.zeros(1, 1, D, 1, dtype=torch.float64)
    row_m = torch.zeros(1, 1, D, dtype=torch.bool)
    row_m[..., list(cols)] = True
    out["context_data"] = torch.cat([inp["context_data"], row_x], dim=1)
    out["context_int_mask"] = torch.cat([inp["context_int_mask"], row_m], dim=1)
    return out


@pytest.mark.parametrize("cols, slot", [((), 0), ((0,), 1)])
def test_real_zeros(cols, slot):
    """An all-zero row is data: it changes predictions and n_r."""
    m = model()
    inp, _ = batch([[((), 7), ((0,), 5)]])
    more = _append_row(inp, 0, cols)
    n0 = group_regimes(inp["context_int_mask"]).n_r
    n1 = group_regimes(more["context_int_mask"]).n_r
    assert n1[0, slot] == n0[0, slot] + 1 and n1.sum() == n0.sum() + 1
    assert not torch.allclose(predict(m, **inp), predict(m, **more), atol=1e-6)


# --------------------------------------------------------------------------- #
# Empty slots, isolation, missing regimes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale", ["none", "stats"])
def test_empty_slots_finite(scale):
    m = model(scale_conditioning=scale).train()
    inp, _ = batch(scale=scale == "stats")
    out = m(**inp)
    for t in (out.pred_mean, out.pred_std, out.weights):
        assert torch.isfinite(t).all()
    loss = m.calculate_loss(out, inp["target_data"], inp["outcome_index"])
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_empty_slot_guard():
    """No pooling-mask row is all False (torch >= 2.5 SDPA would hide a NaN
    here by returning zeros, so check the mask itself); present slots read
    exactly their own rows."""
    g = group_regimes(batch(MISSING)[0]["context_int_mask"])
    K = 3
    mask = pooling_mask(g, K)  # [B, R * K, Nc]
    assert mask.any(-1).all()
    member = g.slot_member.repeat_interleave(K, dim=1)
    present = g.slot_present.repeat_interleave(K, dim=1)
    assert torch.equal(mask[present], member[present])


def test_regime_isolation():
    """Block 1: changing regime A's rows leaves regime B's step-2 output and
    step-3 memories unchanged."""
    m = model()
    block = m.encoder.blocks[0]
    inp, lab = batch([[((), 4), ((0,), 5), ((2,), 3)]])
    rows_a = lab[0] == 1  # do(X1), slot 1
    store = {}
    handles = [hook(block.sample_attn, store, "s"), hook(block.pool, store, "m")]
    predict(m, **inp)
    s0, m0 = store["s"], store["m"]  # [1, D, Nc, d], [1, D, R, K, d]
    pert = dict(inp)
    pert["context_data"] = inp["context_data"].clone()
    pert["context_data"][0, rows_a] += 3.0
    predict(m, **pert)
    for h in handles:
        h.remove()
    s1, m1 = store["s"], store["m"]
    assert not torch.allclose(s0[0][:, rows_a], s1[0][:, rows_a], atol=1e-6)
    assert torch.allclose(s0[0][:, ~rows_a], s1[0][:, ~rows_a], **TOL)
    assert not torch.allclose(m0[0, :, 1], m1[0, :, 1], atol=1e-6)
    for r in (0, 2):
        assert torch.allclose(m0[0, :, r], m1[0, :, r], **TOL)


def test_missing_regimes():
    m = model()
    inp, _ = batch(MISSING)
    store = {}
    handles = [hook(b.pool, store, i) for i, b in enumerate(m.encoder.blocks)]
    out = predict(m, **inp)
    for h in handles:
        h.remove()
    assert torch.isfinite(out).all()
    want = torch.tensor([1, 2, 3, 1])
    g = group_regimes(inp["context_int_mask"])
    assert torch.equal(g.slot_present.sum(-1), want)
    for mem in store.values():  # [B, D, R, K, d]
        nonzero = mem.abs().sum(dim=(1, 3, 4)) > 0  # [B, R]
        assert torch.equal(nonzero, g.slot_present)
        assert torch.equal(nonzero.sum(-1), want)


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def test_query_leakage():
    m = model()
    inp, _ = batch()
    ref = predict(m, **inp)
    # Hidden query values (everything but the treatment column) are ignored.
    hidden = dict(inp)
    tgt = inp["target_data"].clone()
    B = tgt.shape[0]
    keep = torch.zeros_like(tgt, dtype=torch.bool)
    keep[torch.arange(B), :, inp["intervention_index"]] = True
    hidden["target_data"] = torch.where(keep, tgt, torch.randn_like(tgt) * 5)
    assert torch.allclose(predict(m, **hidden), ref, **TOL)
    # The treatment value is used.
    moved = dict(inp)
    moved["target_data"] = torch.where(keep, tgt + 1.0, tgt)
    assert not torch.allclose(predict(m, **moved), ref, atol=1e-6)
    # Adding a query row leaves the existing predictions unchanged.
    extra = dict(inp)
    extra["target_data"] = torch.cat([tgt, torch.randn_like(tgt[:, :1])], dim=1)
    assert torch.allclose(predict(m, **extra)[:, :NQ], ref, **TOL)


# --------------------------------------------------------------------------- #
# Flag, count feature, scale conditioning
# --------------------------------------------------------------------------- #


def _flip_target(inp, lab):
    """Regime do(X1) becomes do(X2): same rows, same values."""
    out = dict(inp)
    mask = inp["context_int_mask"].clone()
    rows = lab[0] == 1
    mask[0, rows, 0] = False
    mask[0, rows, 1] = True
    out["context_int_mask"] = mask
    return out


def test_flag_is_used():
    inp, lab = batch([[((), 7), ((0,), 5)]])
    flipped = _flip_target(inp, lab)
    m = model()
    assert not torch.allclose(predict(m, **inp), predict(m, **flipped), atol=1e-6)
    none = model(reg_axis_target_flag="none")
    assert not [n for n, _ in none.named_parameters() if "flag" in n]
    # Without a flag only the grouping is seen, and it is the same partition.
    assert torch.allclose(predict(none, **inp), predict(none, **flipped), **TOL)


@pytest.mark.parametrize("count", [True, False])
def test_count_feature_is_used(count):
    inp, lab = batch([[((), 7), ((0,), 5)]])
    rows = lab[0] == 1
    dup = dict(inp)
    dup["context_data"] = torch.cat(
        [inp["context_data"], inp["context_data"][:, rows]], dim=1
    )
    dup["context_int_mask"] = torch.cat(
        [inp["context_int_mask"], inp["context_int_mask"][:, rows]], dim=1
    )
    m = model(reg_axis_count_feature=count)
    has_count = any(".pool.count." in n for n, _ in m.named_parameters())
    assert has_count == count
    same = torch.allclose(predict(m, **inp), predict(m, **dup), **TOL)
    assert same != count


def test_scale_conditioning():
    inp, _ = batch(scale=True)
    stats = model(scale_conditioning="stats")
    no_scale = {k: v for k, v in inp.items() if k != "context_scale"}
    with pytest.raises(ValueError, match="context_scale"):
        stats(**no_scale)
    other = dict(inp, context_scale=inp["context_scale"] + 1.0)
    assert not torch.allclose(predict(stats, **inp), predict(stats, **other))
    none = model(scale_conditioning="none")
    assert torch.allclose(predict(none, **inp), predict(none, **no_scale), **TOL)
    assert not [n for n, _ in none.named_parameters() if "scale" in n]


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #


def _backward(m, inp):
    m.train()
    out = m(**inp)
    m.calculate_loss(out, inp["target_data"], inp["outcome_index"]).backward()


def test_gradients_reach():
    m = model(scale_conditioning="stats")
    _backward(m, batch(scale=True)[0])

    def nonzero(module):
        return any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )

    for i, block in enumerate(m.encoder.blocks):
        assert block.pool.q_mem.grad.abs().sum() > 0, i
        assert nonzero(block.regime_attn), i
        assert nonzero(block.pool.count), i
        assert nonzero(block.query_read), i
        if block.feedback is not None:
            assert nonzero(block.feedback), i
        assert nonzero(m.encoder.flag.inject_var[i]), i
        assert nonzero(m.encoder.flag.inject_sample[i]), i
    assert nonzero(m.encoder.flag.flag_code)
    assert nonzero(m.encoder.flag.flag_embed)
    assert nonzero(m.encoder.scale_embedding)


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"scale_conditioning": "stats"},
        {"reg_axis_target_flag": "none", "reg_axis_count_feature": False},
        {"reg_axis_num_blocks": 1, "reg_axis_num_memory_tokens": 1},
    ],
)
def test_no_unused_params(over):
    m = model(randomise=False, **over)  # default init
    _backward(m, batch(scale=True)[0])
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def test_checkpoint_roundtrip(tmp_path):
    over = dict(
        scale_conditioning="stats",
        reg_axis_num_blocks=3,
        reg_axis_num_memory_tokens=2,
        reg_axis_count_feature=False,
    )
    m = model(**over).float()
    c = tiny_config()
    resolved = {
        "d_model": c.d_model,
        "emb_depth": c.emb_depth,
        "dim_feedforward": c.dim_feedforward,
        "nhead": c.nhead,
        "num_layers_encoder": c.num_layers_encoder,
        "sample_attn_mode": c.sample_attn_mode,
        "num_mixture_components": c.num_mixture_components,
        "model_class": "RegimeAxisMixedContextMoGModel",
        "intervention_encoding": "reg_axis",
        **over,
    }
    yml = tmp_path / "resolved_model_config.yaml"
    yml.write_text(yaml.safe_dump(resolved))
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": m.state_dict()}, ckpt)
    loaded = load_model(str(ckpt), str(yml), "cpu", num_nodes=D)
    assert type(loaded) is RegimeAxisMixedContextMoGModel
    assert len(loaded.encoder.blocks) == 3
    inp, _ = batch(scale=True)
    inp = {k: v.float() if v.is_floating_point() else v for k, v in inp.items()}
    assert torch.equal(predict(m, **inp), predict(loaded, **inp))
    # Cross-loading between encodings fails.
    lw = LayerwiseMixedContextMoGModel(tiny_config())
    with pytest.raises(RuntimeError):
        lw.load_state_dict(model().state_dict())
    with pytest.raises(RuntimeError):
        model().load_state_dict(lw.state_dict())
