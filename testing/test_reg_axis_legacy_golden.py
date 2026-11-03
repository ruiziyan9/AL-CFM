"""Legacy encodings (r1, layerwise, extra_node) are unchanged by reg_axis.

The fixture was written by make_reg_axis_legacy_golden.py on the commit
before any reg_axis code existed. Each state dict must load strictly into a
freshly built model and reproduce the recorded outputs.
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_mixed_context_model import tiny_config  # noqa: E402

from CITNP.models.mixed_context_model import INTERVENTION_ENCODINGS  # noqa: E402

FIXTURE = HERE / "fixtures" / "reg_axis_legacy_golden.pt"
CASES = [
    (m, s) for m in ("r1", "layerwise", "extra_node") for s in ("none", "stats")
]


def unpack(packed: dict) -> dict:
    """Inverse of make_reg_axis_legacy_golden.pack (float32 tensors)."""
    sd, i, flat = {}, 0, packed["flat"].float()
    for k, shape in packed["keys"]:
        n = int(torch.Size(shape).numel())
        sd[k] = flat[i : i + n].reshape(shape).clone()
        i += n
    assert i == flat.numel()
    return sd


@pytest.fixture(scope="module")
def golden():
    return torch.load(FIXTURE, map_location="cpu")


@pytest.mark.parametrize("method, scale", CASES)
def test_legacy_golden(golden, method, scale):
    case = golden["cases"][f"{method}/{scale}"]
    sd = unpack(case["state_dict"])
    if scale != "none":  # stats cases store only the scale-embedding keys
        sd = {**unpack(golden["cases"][f"{method}/none"]["state_dict"]), **sd}
    cfg = tiny_config()
    cfg.scale_conditioning = scale
    model = INTERVENTION_ENCODINGS[method](cfg).eval()
    model.load_state_dict(sd, strict=True)
    with torch.no_grad():
        out = model(**golden["inputs"])
    for key, ref in case["outputs"].items():
        got = getattr(out, key)
        assert got.shape == ref.shape, key
        assert torch.allclose(got, ref, atol=1e-6, rtol=0), (key, method, scale)
