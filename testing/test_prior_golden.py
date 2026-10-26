"""Regression test: the task generator and the exact oracle reproduce the
outputs recorded before the prior refactor.

Golden files are recorded once with `testing/make_prior_golden.py` on the
original code; never regenerate them to make this test pass.

    pytest testing/test_prior_golden.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_golden_spec import (
    GENERATOR_SETTINGS,
    GOLDEN,
    generator_arrays,
    oracle_arrays,
)


def _load(name: str) -> dict[str, np.ndarray]:
    path = GOLDEN / name
    if not path.exists():
        pytest.fail(
            f"{path} is missing: record it with testing/make_prior_golden.py "
            "on the pre-refactor commit"
        )
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("name", list(GENERATOR_SETTINGS))
def test_generator_matches_golden(name):
    gold = _load(f"generator_{name}.npz")
    new = generator_arrays(name)
    assert set(new) == set(gold)
    for k in gold:
        assert new[k].dtype == gold[k].dtype, k
        np.testing.assert_array_equal(new[k], gold[k], err_msg=k)


def test_oracle_matches_golden():
    gold = _load("oracle.npz")
    new = oracle_arrays()
    assert set(new) == set(gold)
    np.testing.assert_array_equal(new["dags"], gold["dags"])
    for k in gold:
        if k != "dags":
            np.testing.assert_allclose(
                new[k], gold[k], rtol=0, atol=1e-10, err_msg=k
            )
