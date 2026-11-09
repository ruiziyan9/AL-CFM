"""Backward compatibility of model_api after the flag-propagation extension:
default `context_states` and `predict_mixture` reproduce the outputs recorded
with the code from before the edit (testing/make_model_api_states_golden.py).

Comparisons use tight tolerances so they hold across machines and torch / BLAS
builds; the bit-for-bit variant is opt-in (`-m golden_exact`) for
same-machine regression checks.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from flag_propagation_spec import (  # noqa: E402
    ARCHITECTURES,
    GOLDEN_MODEL,
    build_model,
    parameter_checksum,
)

from CITNP.evaluation.model_api import (  # noqa: E402
    context_states,
    hidden_state_layout,
    predict_mixture,
)

GOLDEN = HERE / "fixtures" / "model_api_states_golden.pt"
# float64 outputs (predict_mixture, parameters); float32 outputs
# (context_states returns float32 by contract)
TOL64 = dict(rtol=1e-10, atol=1e-12)
TOL32 = dict(rtol=1e-5, atol=1e-6)
CASES = [(a, m) for a in ARCHITECTURES for m in ("MHCA", "MHSA")]


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN, map_location="cpu", weights_only=False)


def environment_note(golden) -> str:
    here = dict(torch_version=torch.__version__, device="cpu", dtype="torch.float64")
    return f"golden recorded with {golden['environment']}; running with {here}"


def outputs(golden, arch, mode):
    with tempfile.TemporaryDirectory() as tmp:
        model = build_model(Path(tmp), arch, mode=mode, dtype=torch.float64,
                            **GOLDEN_MODEL)  # fmt: skip
    args = (model, golden["ctx_x"], golden["ctx_int"], golden["T"], golden["Y"])
    out = {"parameter_checksum": parameter_checksum(model)}
    for zero in (False, True):
        out[f"states_zero{int(zero)}"] = context_states(*args, device="cpu", zero_flags=zero)  # fmt: skip
        out[f"mixture_zero{int(zero)}"] = predict_mixture(
            *args, golden["grid"], device="cpu", zero_flags=zero
        )
    return out


def compare(got, want, note, exact=False):
    def close(a, b, what):
        if exact:
            assert torch.equal(a, b), f"{what} differs bit for bit; {note}"
        else:
            tol = TOL64 if b.dtype == torch.float64 else TOL32
            torch.testing.assert_close(a, b, **tol, msg=lambda m: f"{what}: {m}\n{note}")

    close(got["parameter_checksum"], want["parameter_checksum"], "parameters")
    for zero in (0, 1):
        g, w = got[f"states_zero{zero}"], want[f"states_zero{zero}"]
        assert list(g) == list(w)
        for name in w:
            assert g[name].shape == w[name].shape and g[name].dtype == torch.float32
            close(g[name], w[name], f"context_states[{name}] zero_flags={zero}")
        for a, b in zip(got[f"mixture_zero{zero}"], want[f"mixture_zero{zero}"]):
            close(a, b, f"predict_mixture zero_flags={zero}")


@pytest.mark.parametrize("arch, mode", CASES)
def test_defaults_match_golden(golden, arch, mode):
    compare(outputs(golden, arch, mode), golden["models"][f"{arch}/{mode}"],
            environment_note(golden))  # fmt: skip


@pytest.mark.golden_exact
@pytest.mark.parametrize("arch, mode", CASES)
def test_defaults_match_golden_bit_for_bit(golden, arch, mode):
    compare(outputs(golden, arch, mode), golden["models"][f"{arch}/{mode}"],
            environment_note(golden), exact=True)  # fmt: skip


def test_fixture_is_small():
    assert GOLDEN.stat().st_size < 512 * 1024


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
