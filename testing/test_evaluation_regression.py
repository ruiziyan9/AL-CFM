"""Regression test: the evaluation code reproduces the pre-refactor outputs.

Golden files are recorded once with `testing/make_golden.py` on the original
code; never regenerate them to make this test pass. The trained-checkpoint
part runs only when $ALCFM_TRAINED_CKPT is set.

    pytest testing/test_evaluation_regression.py
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import (  # noqa: E402
    BANK_ARGS,
    DATA_CFG,
    GOLDEN,
    KSHOT_ARGS,
    REPO,
    SUITE_ARGS,
    TINY_CFG,
    TRAINED_CFG,
    TRAINED_ENV,
    callback_bank_gap,
    wrapper_outputs,
)

RTOL, ATOL = 1e-5, 1e-7

pytestmark = pytest.mark.skipif(
    not (GOLDEN / "tiny_model.pt").exists(),
    reason="golden files not recorded (run testing/make_golden.py)",
)


# --------------------------------------------------------------------------- #
# Entry points under test (the only part that changes during the refactor)
# --------------------------------------------------------------------------- #
def _run_main(main, args: list[str]) -> None:
    argv = sys.argv
    sys.argv = ["prog", *args]
    try:
        main()
    finally:
        sys.argv = argv


def run_kshot(args: list[str]) -> None:
    from CITNP.evaluation.eval_d3_kshot import main

    _run_main(main, args)


def run_suite(args: list[str]) -> None:
    from CITNP.evaluation.eval_d3_oracle import main

    _run_main(main, args)


def run_bank(args: list[str]) -> None:
    from CITNP.evaluation.oracle_bank import main

    _run_main(main, args)


def load(ckpt: str, model_cfg: str):
    from CITNP.evaluation.eval_d3_oracle import load_model

    return load_model(ckpt, model_cfg, "cpu")


def wrapper_fns():
    """(mixtures_fn, logpdf_fn, mog_fn) with the signatures in golden_spec."""
    from CITNP.evaluation.eval_d3_kshot import model_mixtures
    from CITNP.evaluation.eval_d3_oracle import model_logpdf
    from CITNP.trainer.monitor_callbacks import mog_predictive

    return model_mixtures, model_logpdf, mog_predictive


# --------------------------------------------------------------------------- #
# Fixtures and comparison helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _in_repo(monkeypatch):
    monkeypatch.chdir(REPO)


@pytest.fixture(params=["tiny", "trained"])
def model_set(request) -> tuple[Path, str, str]:
    """(golden dir, checkpoint, model config)."""
    if request.param == "tiny":
        return GOLDEN, str(GOLDEN / "tiny_model.pt"), TINY_CFG
    ckpt = os.environ.get(TRAINED_ENV)
    if not ckpt:
        pytest.skip(f"${TRAINED_ENV} not set")
    if not (GOLDEN / "trained").exists():
        pytest.skip("no golden files for the trained checkpoint")
    return GOLDEN / "trained", ckpt, TRAINED_CFG


@pytest.fixture(scope="module")
def bank_items() -> list[dict]:
    with open(GOLDEN / "bank.pkl", "rb") as f:
        return pickle.load(f)["items"]


def assert_frames_match(new: pd.DataFrame, old: pd.DataFrame) -> None:
    assert list(new.columns) == list(old.columns)
    assert len(new) == len(old)
    for col in old.columns:
        if pd.api.types.is_float_dtype(old[col]):
            np.testing.assert_allclose(
                new[col].to_numpy(), old[col].to_numpy(), rtol=RTOL, atol=ATOL,
                err_msg=col,
            )  # fmt: skip
        else:
            np.testing.assert_array_equal(
                new[col].to_numpy(), old[col].to_numpy(), err_msg=col
            )


def assert_values_equal(new, old, path: str = "") -> None:
    """Exact, recursive equality of bank contents."""
    assert type(new) is type(old), path
    if isinstance(old, dict):
        assert list(new) == list(old), path
        for k in old:
            assert_values_equal(new[k], old[k], f"{path}.{k}")
    elif isinstance(old, (list, tuple)):
        assert len(new) == len(old), path
        for i, (n, o) in enumerate(zip(new, old)):
            assert_values_equal(n, o, f"{path}[{i}]")
    elif isinstance(old, np.ndarray):
        assert new.dtype == old.dtype and new.shape == old.shape, path
        np.testing.assert_array_equal(new, old, err_msg=path)
    else:
        assert new == old, path


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_kshot(model_set, tmp_path):
    gdir, ckpt, cfg = model_set
    out = tmp_path / "kshot.csv"
    run_kshot(
        ["--ckpt", ckpt, "--model_config", cfg, "--data_config", DATA_CFG,
         *KSHOT_ARGS, "--out", str(out)]
    )  # fmt: skip
    assert_frames_match(pd.read_csv(out), pd.read_csv(gdir / "kshot.csv"))
    assert_frames_match(
        pd.read_csv(tmp_path / "kshot_by_t.csv"), pd.read_csv(gdir / "kshot_by_t.csv")
    )


def test_scm_suite(model_set, tmp_path):
    gdir, ckpt, cfg = model_set
    out = tmp_path / "suite.csv"
    run_suite(
        ["--ckpt", ckpt, "--model_config", cfg, "--data_config", DATA_CFG,
         *SUITE_ARGS, "--out", str(out)]
    )  # fmt: skip
    assert_frames_match(pd.read_csv(out), pd.read_csv(gdir / "suite.csv"))


def test_bank(tmp_path):
    out = tmp_path / "bank.pkl"
    run_bank(["--data_config", DATA_CFG, "--out", str(out), *BANK_ARGS])
    with open(out, "rb") as f:
        new = pickle.load(f)
    with open(GOLDEN / "bank.pkl", "rb") as f:
        old = pickle.load(f)
    assert_values_equal(new, old)


def test_bank_gap(model_set, bank_items, tmp_path):
    gdir, ckpt, cfg = model_set
    model = load(ckpt, cfg)
    with open(gdir / "bank_gap.json") as f:
        old = json.load(f)
    for norm in (True, False):
        new = callback_bank_gap(model, bank_items, norm, tmp_path)
        ref = old[f"normalise={norm}"]
        assert list(new) == list(ref)
        np.testing.assert_allclose(
            [new[k] for k in ref], [ref[k] for k in ref], rtol=RTOL, atol=ATOL
        )


def test_wrappers(model_set, bank_items):
    gdir, ckpt, cfg = model_set
    model = load(ckpt, cfg)
    new = wrapper_outputs(model, bank_items, *wrapper_fns())
    old = np.load(gdir / "wrappers.npz")
    assert sorted(new) == sorted(old.files)
    for k in old.files:
        np.testing.assert_allclose(new[k], old[k], rtol=RTOL, atol=ATOL, err_msg=k)
