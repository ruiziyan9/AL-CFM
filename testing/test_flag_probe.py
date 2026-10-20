"""Tests for the flag probe: model_api.context_states, probes, run_flag_probe
and the `probe-flag` command (random tiny model, CPU)."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import D, TINY_CFG, build_tiny_model

from CITNP.evaluation.experiments import run_flag_probe
from CITNP.evaluation.model_api import (
    context_states,
    predict_mixture,
    probe_points,
)
from CITNP.evaluation.probes import balanced_accuracy, fit_linear_probe, roc_auc

N = 40


def fake_bank(n_contexts=12, seed=0) -> list[dict]:
    """Contexts with observational rows and do-rows on one random column."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_contexts):
        x = rng.normal(size=(N, D)).astype(np.float32)
        ci = np.zeros((N, D), dtype=bool)
        ci[N // 2 :, rng.integers(D)] = True
        T, Y = rng.choice(D, 2, replace=False)
        items.append(dict(ctx_x=x, ctx_int=ci, T=int(T), Y=int(Y)))
    return items


def model_with_flag(seed=7):
    """Tiny R1 model with a non-zero (randomised) flag embedding."""
    model = build_tiny_model().eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.encoder.context_intervened.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    return model


def test_context_states_shapes_and_names():
    model = build_tiny_model().eval()
    it = fake_bank(1)[0]
    st = context_states(model, it["ctx_x"], it["ctx_int"], it["T"], it["Y"],
                        device="cpu")  # fmt: skip
    names = probe_points(model)
    n_blocks = len(model.encoder.encoder.node_layers)
    assert list(st) == names and len(names) == 1 + 2 * n_blocks
    for v in st.values():
        assert v.shape == (N, D, model.d_model)


def test_states_are_the_ones_the_model_uses():
    """Hooks must not change the forward pass."""
    model = model_with_flag()
    it = fake_bank(1)[0]
    args = (model, it["ctx_x"], it["ctx_int"], it["T"], it["Y"])
    t = np.linspace(-2, 2, 5)
    before = predict_mixture(*args, t, device="cpu")
    context_states(*args, device="cpu")
    after = predict_mixture(*args, t, device="cpu")
    for a, b in zip(before, after):
        assert torch.equal(a, b)


def test_zero_init_flag_changes_nothing_and_trained_flag_does():
    it = fake_bank(1)[0]
    args = (it["ctx_x"], it["ctx_int"], it["T"], it["Y"])
    fresh = build_tiny_model().eval()  # flag MLP last layer is zero
    on = context_states(fresh, *args, device="cpu")
    off = context_states(fresh, *args, device="cpu", zero_flags=True)
    assert all(torch.equal(on[k], off[k]) for k in on)

    model = model_with_flag()
    on = context_states(model, *args, device="cpu")
    off = context_states(model, *args, device="cpu", zero_flags=True)
    ci = torch.as_tensor(it["ctx_int"])
    diff = (on["embed"] - off["embed"]).abs().sum(-1)  # [N, D]
    assert (diff[ci] > 0).all() and (diff[~ci] == 0).all()


def test_probe_metrics():
    assert (
        roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    )
    assert (
        balanced_accuracy(np.array([-1.0, 1.0, 1.0]), np.array([0, 1, 0])) == 0.75
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5))
    groups = np.repeat(np.arange(30), 20)
    y = X[:, 0] + 0.1 * rng.normal(size=600) > 0
    assert fit_linear_probe(X, y, groups)["auc"] > 0.95
    noise = fit_linear_probe(X, rng.random(600) > 0.5, groups)
    assert abs(noise["auc"] - 0.5) < 0.1


def test_run_flag_probe_table():
    model = model_with_flag()
    df = run_flag_probe(model, fake_bank(), device="cpu", n_folds=3)
    names = probe_points(model)
    assert len(df) == 1 + 2 * len(names)
    assert set(df.columns) >= {"point", "depth", "condition", "auc", "auc_se",
                               "bal_acc", "rel_change_int", "rel_change_obs"}  # fmt: skip
    on = df[df["condition"] == "flags_on"].set_index("point")
    # the flag is added at the embedding, so it is decodable there
    assert on.loc["embed", "auc"] > 0.8
    assert on.loc["embed", "rel_change_obs"] == 0


@pytest.mark.parametrize("plot_only", [False, True])
def test_probe_flag_cli(tmp_path, plot_only):
    from CITNP.evaluation.__main__ import main

    ckpt, bank = tmp_path / "m.pt", tmp_path / "bank.pkl"
    torch.save({"model_state_dict": model_with_flag().state_dict()}, ckpt)
    with open(bank, "wb") as f:
        pickle.dump({"items": fake_bank()}, f)
    out = tmp_path / "probe.csv"
    main(["probe-flag", "--ckpt", str(ckpt), "--model_config", TINY_CFG,
          "--bank", str(bank), "--n_folds", "3", "--device", "cpu",
          "--out", str(out)])  # fmt: skip
    if plot_only:
        (tmp_path / "probe.png").unlink()
        main(["probe-flag", "--plot_only", str(out)])
    assert len(pd.read_csv(out)) > 0
    assert (tmp_path / "probe.png").exists()
