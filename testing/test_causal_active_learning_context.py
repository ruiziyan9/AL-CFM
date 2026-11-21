"""Contexts, candidate sets and the constrained environment (plan tests 1, 8,
9; core)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_active_learning.acquisitions import (
    Acquisition,
    ConstrainedEnvironment,
    NonIntervenableNodeError,
    build_candidate_set,
)
from causal_active_learning.context import ExperimentalContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import StubEnvironment, empty_context

DIMENSIONS = [2, 3, 5, 8]


@pytest.mark.parametrize("D", DIMENSIONS)
@pytest.mark.parametrize("R", [1, 3])
def test_context_append(D, R):
    context = empty_context(rows=15, D=D)
    rows = torch.randn(R, D, dtype=torch.float64)
    node = D - 1
    after = context.append(rows, node)
    assert after.values.shape == (15 + R, D)
    assert torch.equal(after.values[:15], context.values)
    assert torch.equal(after.intervened[:15], context.intervened)
    assert after.intervened[15:, node].all() and after.intervened.sum() == R
    assert after.newest_regime_size() == R
    observational = context.append(rows, None)
    assert not observational.intervened.any()
    assert observational.newest_regime_size() is None
    again = after.append(rows, node)
    assert again.newest_regime_size() == 2 * R


def test_context_validation():
    with pytest.raises(ValueError):
        ExperimentalContext(
            torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.bool)
        )
    with pytest.raises(ValueError):
        ExperimentalContext(
            torch.tensor([[float("nan"), 0.0]]),
            torch.zeros(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError):  # two intervened nodes in one row
        ExperimentalContext(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))


@pytest.mark.parametrize("D", DIMENSIONS)
def test_build_candidate_set(D):
    everything = build_candidate_set(D, None, True, True)
    assert everything.intervenable_nodes == tuple(range(D))
    assert everything.observational
    subset = tuple(range(1, D))
    assert build_candidate_set(D, subset, True, True).intervenable_nodes == subset
    # treatment / outcome are candidates iff listed
    assert 0 not in build_candidate_set(D, subset, False, True).intervenable_nodes
    assert not build_candidate_set(D, subset, True, False).observational
    assert not build_candidate_set(D, subset, False, True).observational
    with pytest.raises(ValueError):
        build_candidate_set(D, [D], True, True)
    with pytest.raises(ValueError):
        build_candidate_set(D, [0, 0], True, True)
    with pytest.raises(ValueError, match="include_observational_candidate"):
        build_candidate_set(D, [], True, False)
    assert build_candidate_set(D, [], True, True).observational


def test_constrained_environment():
    env = ConstrainedEnvironment(StubEnvironment(), [1, 3])
    rng = np.random.default_rng(0)
    assert env.run(Acquisition.intervention(1, 0.5, 2), rng).shape == (2, 4)
    assert env.run(Acquisition.observational(2), rng).shape == (2, 4)
    with pytest.raises(NonIntervenableNodeError):
        env.run(Acquisition.intervention(0, 0.5, 2), rng)


def test_acquisition_constructors():
    a = Acquisition.intervention(2, 1.5, 3)
    assert (a.target_node, a.intervention_value, a.num_rows) == (2, 1.5, 3)
    o = Acquisition.observational(3)
    assert o.is_observational and o.num_rows == 3
    with pytest.raises(ValueError):
        Acquisition(1, None)
    with pytest.raises(ValueError):
        Acquisition(None, 1.0)
