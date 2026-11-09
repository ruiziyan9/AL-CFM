"""Backward compatibility of model_api after the flag-propagation extension:
default `context_states` and `predict_mixture` reproduce the outputs recorded
before the edit (testing/make_model_api_states_golden.py) bit for bit."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from flag_propagation_spec import ARCHITECTURES, build_model  # noqa: E402

from CITNP.evaluation.model_api import (  # noqa: E402
    context_states,
    hidden_state_layout,
    predict_mixture,
)

GOLDEN = HERE / "fixtures" / "model_api_states_golden.pt"


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
@pytest.mark.parametrize("mode", ["MHCA", "MHSA"])
def test_defaults_match_golden(golden, arch, mode):
    rec = golden["models"][f"{arch}/{mode}"]
    with tempfile.TemporaryDirectory() as tmp:
        model = build_model(Path(tmp), arch, mode=mode, dtype=torch.float32)
    sd = model.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in rec["state_dict"].items())
    args = (model, golden["ctx_x"], golden["ctx_int"], golden["T"], golden["Y"])
    for zero in (False, True):
        states = context_states(*args, device="cpu", zero_flags=zero)
        want = rec[f"states_zero{int(zero)}"]
        assert list(states) == list(want)
        for name in want:
            assert states[name].shape == want[name].shape
            assert torch.equal(states[name], want[name]), name
        mix = predict_mixture(*args, golden["grid"], device="cpu", zero_flags=zero)
        for a, b in zip(mix, rec[f"mixture_zero{int(zero)}"]):
            assert torch.equal(a, b)


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_extended_states(golden, arch):
    with tempfile.TemporaryDirectory() as tmp:
        model = build_model(Path(tmp), arch, dtype=torch.float32)
    layout = hidden_state_layout(model)
    want_arch, want_share = {
        "r1": ("r1", None),
        "layerwise_separate": ("layerwise", False),
        "layerwise_shared": ("layerwise", True),
        "extra_node": ("extra_node", None),
    }[arch]
    assert layout["architecture"] == want_arch
    assert layout["share_intervention_encoder"] == want_share
    x, ci = golden["ctx_x"], golden["ctx_int"]
    N, D = x.shape
    ctx, qry = context_states(
        model, x, ci, 0, 2, device="cpu", include_metadata_token=True,
        include_query_states=True, t=[-1.0, 0.0, 1.0],
    )  # fmt: skip
    plain = context_states(model, x, ci, 0, 2, device="cpu")
    width = D + 1 if layout["metadata_token"] else D
    for name, v in ctx.items():
        assert v.shape == (N, width, model.d_model)
        assert torch.equal(v[:, :D], plain[name])
        assert qry[name].shape == (3, width, model.d_model)
