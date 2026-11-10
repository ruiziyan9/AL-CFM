"""Tests for the reg_axis_full model (CITNP/models/regime_axis_full)."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml  # type: ignore[import-untyped]

from CITNP.evaluation.model_api import load_model
from CITNP.models.mixed_context_model import (
    INTERVENTION_ENCODINGS,
    resolve_model_class,
)
from CITNP.models.regime_axis import (
    RegimeAxisMixedContextMoGModel,
    group_regimes,
)
from CITNP.models.regime_axis_full import (
    RegimeAxisFullEncoder,
    RegimeAxisFullMixedContextMoGModel,
    pack_regimes,
)
from CITNP.models.regime_axis_full.attention import TwoStageRegimeAttentionLayer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import tiny_config
from test_reg_axis_grouping import build_mask

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
ZERO_INIT = ("encoder.flag.", "encoder.scale_embedding.")


def model(seed=0, randomise=True, **cfg_over):
    cfg = tiny_config()
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    torch.manual_seed(seed)
    m = RegimeAxisFullMixedContextMoGModel(cfg).double().eval()
    if randomise:  # zero-initialised flag / scale / e_own get non-zero values
        g = torch.Generator().manual_seed(seed + 100)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name.startswith(ZERO_INIT) or name.endswith(".e_own"):
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


def task(inp, b):
    """The inputs of task b alone."""
    return {k: v[b : b + 1] for k, v in inp.items()}


def duplicate_regime(inp, rows):
    """Append a copy of the context rows `rows` ([Nc] bool) to every task."""
    out = dict(inp)
    for key in ("context_data", "context_int_mask"):
        out[key] = torch.cat([inp[key], inp[key][:, rows]], dim=1)
    return out


# --------------------------------------------------------------------------- #
# Structure and registry
# --------------------------------------------------------------------------- #


def test_registry_and_structure():
    cls, name = resolve_model_class("reg_axis_full")
    assert cls is RegimeAxisFullMixedContextMoGModel and name == "reg_axis_full"
    assert INTERVENTION_ENCODINGS["reg_axis_full"] is cls
    assert resolve_model_class(None, "RegimeAxisFullMixedContextModel")[1] == (
        "RegimeAxisFullMixedContextModel"
    )
    m = model()
    assert isinstance(m.encoder, RegimeAxisFullEncoder)
    assert not hasattr(m.encoder, "encoder")  # upstream stack removed
    # default depth = num_layers_encoder // 2
    assert len(m.encoder.blocks) == tiny_config().num_layers_encoder // 2
    assert len(model(reg_axis_num_blocks=3).encoder.blocks) == 3
    for block in m.encoder.blocks:  # every block has both reads
        assert isinstance(block.context_read, TwoStageRegimeAttentionLayer)
        assert isinstance(block.query_read, TwoStageRegimeAttentionLayer)
        assert block.context_read is not block.query_read
        assert block.context_read.e_own is not None
        assert block.query_read.e_own is None
    # e_own starts at zero
    fresh = model(randomise=False)
    assert all((b.context_read.e_own == 0).all() for b in fresh.encoder.blocks)
    # role MLPs keep the upstream parameter names
    for role in ("context", "target"):
        for kind in ("variable", "intervention", "outcome"):
            assert f"encoder.{role}_{kind}.0.weight" in m.state_dict()
    # no memory tokens, pooling or within-regime sample attention
    assert not [n for n in m.state_dict() if "q_mem" in n or "pool" in n]


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
# The two-stage layer equals a naive reference
# --------------------------------------------------------------------------- #


def naive_two_stage(layer, x, context, slot, reader_slot):
    """Explicit loops over tasks, variables, readers, regimes and heads.

    x [B, L, D, d] readers, context [B, Nc, D, d], slot [B, Nc] regime of
    each context row, reader_slot [B, L] own regime of each reader or None.
    """
    B, L, D, d = x.shape
    H = layer.nhead
    e = d // H
    xn, cn = layer.norm1(x), layer.norm1(context)
    out = torch.zeros_like(x)
    for b in range(B):
        regimes = sorted(set(slot[b].tolist()))
        for j in range(D):
            for i in range(L):
                reader = xn[b, i, j]
                msgs = []
                for s in regimes:
                    rows = (slot[b] == s).nonzero().flatten()
                    keys = cn[b, rows, j]  # [n_s, d]
                    q, k, v = (
                        layer.to_q(reader),
                        layer.to_k(keys),
                        layer.to_v(keys),
                    )
                    heads = []
                    for h in range(H):
                        sl = slice(h * e, (h + 1) * e)
                        alpha = torch.softmax(k[:, sl] @ q[sl] / math.sqrt(e), 0)
                        heads.append(alpha @ v[:, sl])
                    msg = torch.cat(heads)
                    if layer.count is not None:
                        n = torch.tensor([float(len(rows))], dtype=x.dtype)
                        msg = msg + layer.count(torch.log1p(n))
                    if reader_slot is not None and reader_slot[b, i] == s:
                        msg = msg + layer.e_own
                    msgs.append(msg)
                msgs = torch.stack(msgs)  # [R_b, d]
                q2 = layer.regime_q(reader)
                k2 = layer.regime_k(layer.regime_norm(msgs))
                heads = []
                for h in range(H):
                    sl = slice(h * e, (h + 1) * e)
                    w = torch.softmax(k2[:, sl] @ q2[sl] / math.sqrt(e), 0)
                    heads.append(w @ msgs[:, sl])
                out[b, i, j] = layer.to_out(torch.cat(heads))
    x = x + out
    return x + layer.ff_block(layer.norm2(x))


@pytest.mark.parametrize("count", [True, False])
def test_two_stage_equals_naive(count):
    torch.manual_seed(0)
    d, H = 8, 2
    mask, _ = build_mask(MISSING, D=D, seed=4)  # absent regimes, unequal sizes
    grouping = group_regimes(mask)
    packing = pack_regimes(grouping)
    B, Nc, _ = mask.shape
    context = torch.randn(B, Nc, D, d, dtype=torch.float64)
    queries = torch.randn(B, NQ, D, d, dtype=torch.float64)
    with torch.no_grad():
        # context readers: keys are the readers themselves, own-regime marker
        layer = TwoStageRegimeAttentionLayer(d, H, 16, 0.0, count, True).double()
        layer.e_own.copy_(torch.randn(d, dtype=torch.float64))
        got = layer(context, context, packing, own=packing.own)
        want = naive_two_stage(
            layer, context, context, grouping.slot, grouping.slot
        )
        assert torch.allclose(got, want, **TOL)
        # query readers: no own regime
        layer = TwoStageRegimeAttentionLayer(d, H, 16, 0.0, count, False).double()
        got = layer(queries, context, packing)
        want = naive_two_stage(layer, queries, context, grouping.slot, None)
        assert torch.allclose(got, want, **TOL)
        # messages have the documented shape
        msg = layer.messages(layer.norm1(queries), layer.norm1(context), packing)
        assert msg.shape == (B, D, NQ, packing.num_slots, d)


def test_own_marker_needs_the_parameter():
    layer = TwoStageRegimeAttentionLayer(8, 2, 16, 0.0, True, False).double()
    mask, _ = build_mask(MISSING, D=D)
    packing = pack_regimes(group_regimes(mask))
    x = torch.randn(4, 12, D, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="e_own"):
        layer(x, x, packing, own=packing.own)


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


@pytest.mark.parametrize("count", [False, True])
def test_size_neutrality(count):
    """Without the count feature, duplicating every row of one regime leaves
    all predictions unchanged (attention is an average); with it they change."""
    inp, lab = batch([[((), 7), ((0,), 5)]])
    dup = duplicate_regime(inp, lab[0] == 1)
    m = model(reg_axis_count_feature=count)
    assert not m.training
    has_count = any(".count." in n for n, _ in m.named_parameters())
    assert has_count == count
    same = torch.allclose(predict(m, **inp), predict(m, **dup), **TOL)
    assert same != count


def test_batch_independence():
    """A task's predictions do not depend on the other tasks of the batch,
    which have other numbers of regimes and other regime sizes."""
    m = model(scale_conditioning="stats")
    inp, _ = batch(scale=True)
    together = predict(m, **inp)
    counts = group_regimes(inp["context_int_mask"]).slot_present.sum(-1)
    assert len(set(counts.tolist())) > 1
    for b in range(together.shape[0]):
        assert torch.allclose(
            predict(m, **task(inp, b)), together[b : b + 1], **TOL
        )


# --------------------------------------------------------------------------- #
# Absent regimes, single regime
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale", ["none", "stats"])
def test_absent_regimes_finite(scale):
    m = model(scale_conditioning=scale).train()
    inp, _ = batch(scale=scale == "stats")
    assert not group_regimes(inp["context_int_mask"]).slot_present.all()
    out = m(**inp)
    for t in (out.pred_mean, out.pred_std, out.weights):
        assert torch.isfinite(t).all()
    loss = m.calculate_loss(out, inp["target_data"], inp["outcome_index"])
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_no_all_false_mask_row(monkeypatch):
    """No attention call of a forward pass gets a mask row that is all False
    (torch >= 2.5 SDPA would hide the NaN by returning zeros, so check the
    masks themselves); present regimes read exactly their own rows."""
    inp, _ = batch(MISSING)
    grouping = group_regimes(inp["context_int_mask"])
    packing = pack_regimes(grouping)
    present = grouping.slot_present
    assert packing.key_mask.any(-1).all()
    assert torch.equal(packing.key_mask[present], packing.valid[present])

    masks = []
    sdpa = F.scaled_dot_product_attention

    def recording(*args, **kwargs):
        if kwargs.get("attn_mask") is not None:
            masks.append(kwargs["attn_mask"])
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording)
    m = model()
    predict(m, **inp)
    # two reads per block, two stages per read
    assert len(masks) == 4 * len(m.encoder.blocks)
    R, N = packing.num_slots, packing.max_rows
    assert {tuple(mask.shape[-1:]) for mask in masks} == {(R,), (N,)}
    for mask in masks:
        assert mask.dtype == torch.bool and mask.any(-1).all()


def test_single_regime_task():
    """An observational-only task: one regime, with or without a mask."""
    m = model()
    inp, _ = batch([[((), 12)]])
    out = predict(m, **inp)
    assert torch.isfinite(out).all()
    no_mask = {k: v for k, v in inp.items() if k != "context_int_mask"}
    assert torch.allclose(predict(m, **no_mask), out, **TOL)
    one_row = dict(inp)  # a single context row
    one_row["context_data"] = inp["context_data"][:, :1]
    one_row["context_int_mask"] = inp["context_int_mask"][:, :1]
    assert torch.isfinite(predict(m, **one_row)).all()


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
# Flag, count feature, own-regime marker, scale conditioning
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


def test_count_feature_is_used():
    inp, _ = batch()
    m = model()
    ref = predict(m, **inp)
    with torch.no_grad():
        m.encoder.blocks[0].query_read.count.weight.add_(0.5)
    moved = predict(m, **inp)
    # only tasks with regimes of different sizes can tell the counts apart ...
    assert not torch.allclose(moved, ref, atol=1e-6)
    # ... and the context read uses its own count as well
    with torch.no_grad():
        m.encoder.blocks[0].context_read.count.weight.add_(0.5)
    assert not torch.allclose(predict(m, **inp), moved, atol=1e-6)


def test_own_regime_marker_is_used():
    inp, _ = batch()
    m = model()
    ref = predict(m, **inp)
    multi = group_regimes(inp["context_int_mask"]).slot_present.sum(-1) > 1
    with torch.no_grad():
        for block in m.encoder.blocks:
            block.context_read.e_own.zero_()
    off = predict(m, **inp)
    # With one regime the marker shifts every message equally, which still
    # changes the update; with several it also marks the reader's regime.
    assert not torch.allclose(off[multi], ref[multi], atol=1e-6)
    assert not torch.allclose(off, ref, atol=1e-6)


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
    # Every parameter of the blocks (including the last block's context
    # read), the flag and the scale embedding gets a non-zero gradient. (The
    # first layer of a role MLP can get exactly zero: masked query cells are 0.)
    new = ("encoder.blocks.", "encoder.flag.", "encoder.scale_embedding.")
    checked = 0
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        if name.startswith(new):
            assert p.grad.abs().sum() > 0, name
            checked += 1
    assert checked > 50
    last = m.encoder.blocks[-1]
    assert last.context_read.e_own.grad.abs().sum() > 0
    assert last.context_read.count.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"scale_conditioning": "stats"},
        {"reg_axis_target_flag": "none", "reg_axis_count_feature": False},
        {"reg_axis_num_blocks": 1},
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
        "model_class": "RegimeAxisFullMixedContextMoGModel",
        "intervention_encoding": "reg_axis_full",
        **over,
    }
    yml = tmp_path / "resolved_model_config.yaml"
    yml.write_text(yaml.safe_dump(resolved))
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": m.state_dict()}, ckpt)
    loaded = load_model(str(ckpt), str(yml), "cpu", num_nodes=D)
    assert type(loaded) is RegimeAxisFullMixedContextMoGModel
    assert len(loaded.encoder.blocks) == 3
    inp, _ = batch(scale=True)
    inp = {k: v.float() if v.is_floating_point() else v for k, v in inp.items()}
    assert torch.equal(predict(m, **inp), predict(loaded, **inp))
    # Cross-loading between the two regime-axis encodings fails.
    reg_axis = RegimeAxisMixedContextMoGModel(tiny_config())
    with pytest.raises(RuntimeError):
        reg_axis.load_state_dict(model().state_dict())
    with pytest.raises(RuntimeError):
        model().load_state_dict(reg_axis.state_dict())
