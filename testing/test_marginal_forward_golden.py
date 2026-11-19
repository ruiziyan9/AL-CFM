"""The marginal models of all five encodings are unchanged by the
prediction-head refactor (each family's forward split into
`_encode_query_tokens` and `_predict_from_query_tokens`).

The fixture was written by make_marginal_forward_golden.py on origin/bmrc,
before the refactor. Each model is rebuilt from the same seed on this branch:
the state-dict keys and parameter sums must match, and so must the outputs.
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_marginal_forward_golden import (  # noqa: E402
    ENCODINGS,
    OUT as FIXTURE,
    build,
)


@pytest.fixture(scope="module")
def golden():
    return torch.load(FIXTURE, map_location="cpu")


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_marginal_forward_matches_golden(golden, encoding):
    case = golden["cases"][encoding]
    model = build(encoding)
    sd = model.state_dict()
    assert list(sd) == case["param_names"]
    for k, v in sd.items():
        assert v.double().sum().item() == pytest.approx(
            case["param_sums"][k], abs=1e-9
        ), k
    with torch.no_grad():
        out = model(**golden["inputs"])
    for name, ref in case["outputs"].items():
        got = getattr(out, name)
        assert got.shape == ref.shape, name
        # the legacy-golden tolerance (CPU matmuls differ in the last bits)
        assert torch.allclose(got, ref, atol=1e-6, rtol=0), (encoding, name)
