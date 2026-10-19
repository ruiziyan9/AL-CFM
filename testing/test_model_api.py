"""Unit tests for CITNP/evaluation/model_api.py (random tiny model, CPU)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import GRID, QUERIES, build_tiny_model

from CITNP.evaluation.model_api import (
    effective_components,
    mixture_cdf,
    mixture_logpdf,
    predict_mixture,
)
from CITNP.evaluation.oracle import LinearSCM


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    return build_tiny_model().eval()


@pytest.fixture(scope="module")
def context() -> tuple[np.ndarray, np.ndarray]:
    B = np.zeros((3, 3))
    B[1, 0], B[1, 2], B[0, 2] = 0.9, 0.8, -0.3
    scm = LinearSCM(B, np.array([0.2, 0.3, 0.25]))
    rng = np.random.default_rng(0)
    xo, mo = scm.sample(64, None, None, rng)
    xi, mi = scm.sample(8, 0, rng.normal(0, 1.5, 8), rng)
    return np.concatenate([xo, xi]), np.concatenate([mo, mi])


@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("zero_flags", [False, True])
def test_batch_equals_single(model, context, normalise, zero_flags):
    x, m = context
    Ts = np.array([T for T, _ in QUERIES])
    Ys = np.array([Y for _, Y in QUERIES])
    kw = dict(device="cpu", normalise=normalise, zero_flags=zero_flags)
    batch = predict_mixture(model, x, m, Ts, Ys, GRID, **kw)
    for q, (T, Y) in enumerate(QUERIES):
        single = predict_mixture(model, x, m, T, Y, GRID, **kw)
        for b, s in zip(batch, single):
            assert s.shape == (1, len(GRID), b.shape[-1])
            np.testing.assert_allclose(
                b[q].numpy(), s[0].numpy(), rtol=1e-5, atol=1e-6
            )


def test_outputs(model, context):
    x, m = context
    mu, sd, logw = predict_mixture(model, x, m, 0, 2, GRID, device="cpu")
    assert mu.dtype == sd.dtype == logw.dtype == torch.float64
    np.testing.assert_allclose(torch.logsumexp(logw, -1).numpy(), 0.0, atol=1e-12)
    assert (sd > 0).all()
    y = np.linspace(-2, 2, len(GRID))
    lp = mixture_logpdf(mu, sd, logw, y)
    cdf = mixture_cdf(mu, sd, logw, y)
    eff = effective_components(logw)
    assert lp.shape == cdf.shape == eff.shape == (1, len(GRID))
    assert ((cdf >= 0) & (cdf <= 1)).all()
    assert ((eff >= 1 - 1e-9) & (eff <= logw.shape[-1] + 1e-9)).all()
    # Density integrates to ~1 over y on a wide grid (raw scale incl. Jacobian).
    yy = torch.linspace(-30, 30, 20001, dtype=torch.float64)
    dens = mixture_logpdf(mu[:, :1], sd[:, :1], logw[:, :1], yy[:, None]).exp()
    np.testing.assert_allclose(
        torch.trapezoid(dens[:, 0], yy).item(), 1.0, atol=1e-4
    )


def test_shape_mismatch_raises(model, context):
    x, m = context
    with pytest.raises(ValueError):
        predict_mixture(model, x, m, [0, 1], [2], GRID, device="cpu")
