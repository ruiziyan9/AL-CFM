"""Tests for CITNP/trainer/mixed_context_trainer.py (R1 Part C)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from CITNP.datasets.mixed_regime_generator import (
    MixedRegimeConfig,
    MixedRegimeTaskDataset,
    collate_tasks,
    generate_task,
)
from CITNP.models.causalinferencemodel import MoGCausalInferenceModel
from CITNP.models.mixed_context_model import (
    MixedContextMoGModel,
    load_upstream_checkpoint,
)
from CITNP.trainer.mixed_context_trainer import (
    MixedContextTrainer,
    _normalisation_stats,
    prepare_mixed_batch,
)
from CITNP.utils.configs import CausalInfModelConfig
from CITNP.utils.utils import convert_trainerdict_to_config

D = 3


def grouped_cfg(**kw) -> MixedRegimeConfig:
    base: dict[str, Any] = dict(
        function_generator="linearfixedstd",
        num_variables=[D],
        graph_type=["ER"],
        graph_degrees={D: [1, 2, 3]},
        n_ctx_range=(16, 64),
        n_qry=12,
        qry_mode="grouped",
        qry_groups=[1, 2, 3],
    )
    base.update(kw)
    return MixedRegimeConfig(**base)


def grouped_batch(G=2, B=6, n_ctx=40, cfg=None, seed0=0):
    cfg = cfg or grouped_cfg()
    tasks = [
        generate_task(cfg, seed0 + s, num_variables=D, n_ctx=n_ctx, n_groups=G)
        for s in range(B)
    ]
    return collate_tasks(tasks)


def tiny_model_config() -> CausalInfModelConfig:
    return CausalInfModelConfig(
        d_model=32,
        emb_depth=2,
        dim_feedforward=64,
        nhead=4,
        dropout=0.0,
        num_layers_encoder=4,
        num_nodes=D,
        sample_attn_mode="MHCA",
        linear_attention=False,
        mean_loss_across_samples=False,
        device="cpu",
        dtype="float32",
        num_mixture_components=3,
    )


def prep(batch, training=False, **kw):
    return prepare_mixed_batch(batch, torch.float32, training, **kw)


def test_prepare_real_grouped_batch():
    batch = grouped_batch(G=2)
    out = prep(batch, normalise=False)
    B, G = 6, 2
    assert out["B"] == B and out["G"] == G
    T = out["intervention_index"].view(B, G)
    Y = out["outcome_index"].view(B, G)
    for b in range(B):
        assert T[b].tolist() == batch["meta"][b]["group_treatments"]
    assert (Y != T).all()
    assert ((Y >= 0) & (Y < D)).all()
    # Validation outcomes are deterministic.
    assert torch.equal(
        prep(batch, normalise=False)["outcome_index"], out["outcome_index"]
    )
    # Training outcomes change with the step.
    ys = {
        tuple(prep(batch, training=True, global_step=s)["outcome_index"].tolist())
        for s in range(5)
    }
    assert len(ys) > 1
    same_step = prep(batch, training=True, global_step=3)["outcome_index"]
    assert torch.equal(
        same_step, prep(batch, training=True, global_step=3)["outcome_index"]
    )
    # Every group sees its task's full context.
    ctx = out["context"].view(B, G, *out["context"].shape[1:])
    cint = out["context_int_mask"].view(B, G, *out["context_int_mask"].shape[1:])
    for g in range(G):
        assert torch.allclose(ctx[:, g, ..., 0], batch["ctx_x"])
        assert torch.equal(cint[:, g], batch["ctx_int"])
    # Targets of group g are that group's query rows.
    tgt = out["target"].view(B, G, -1, D)
    assert torch.allclose(tgt, batch["qry_x"].view(B, G, -1, D))
    assert len(out["group_types"]) == B * G
    assert out["group_types"][1] == batch["meta"][0]["group_types"][1]


def test_normalisation_ignores_intervened_cells():
    batch = grouped_batch(G=2)
    # Pick a task with an intervened context cell.
    b, i, d = (int(v) for v in torch.nonzero(batch["ctx_int"])[0])
    mean0, std0 = prep(batch)["norm"]
    batch["ctx_x"][b, i, d] = 1e3
    mean1, std1 = prep(batch)["norm"]
    assert torch.allclose(mean0, mean1) and torch.allclose(std0, std1)
    # The stats are those of the non-intervened cells.
    x = batch["ctx_x"][b, :, d][~batch["ctx_int"][b, :, d]].double()
    assert torch.isclose(mean1[b, d].double(), x.mean(), atol=1e-5)
    assert torch.isclose(std1[b, d].double(), x.std(unbiased=False), atol=1e-5)


def test_normalisation_fallbacks():
    ctx = torch.randn(2, 5, 2)
    cint = torch.zeros(2, 5, 2, dtype=torch.bool)
    cint[0, :, 0] = True  # all intervened -> use all cells
    cint[1, 1:, 1] = True  # one free cell -> use all cells
    mean, std = _normalisation_stats(ctx, cint)
    assert torch.isclose(mean[0, 0], ctx[0, :, 0].mean(), atol=1e-6)
    assert torch.isclose(mean[1, 1], ctx[1, :, 1].mean(), atol=1e-6)
    mean, std = _normalisation_stats(ctx[:, :1], cint[:, :1])  # one row
    assert torch.equal(mean, torch.zeros(2, 2)) and torch.equal(
        std, torch.ones(2, 2)
    )


def test_guards():
    cfg = grouped_cfg()
    padded = collate_tasks(
        [
            generate_task(cfg, 0, n_ctx=20, n_groups=2),
            generate_task(cfg, 1, n_ctx=30, n_groups=2),
        ]
    )
    with pytest.raises(ValueError, match="padding-free"):
        prep(padded)
    per_row = MixedRegimeConfig(
        "linearfixedstd", [D], ["ER"], {D: [1, 2]}, n_ctx_range=(16, 64), n_qry=12
    )
    with pytest.raises(ValueError, match="grouped"):
        prep(
            collate_tasks([generate_task(per_row, s, n_ctx=20) for s in range(2)])
        )


def _upstream_and_mixed():
    torch.manual_seed(0)
    up = MoGCausalInferenceModel(tiny_model_config()).eval()
    mixed = MixedContextMoGModel(tiny_model_config()).eval()
    load_upstream_checkpoint(mixed, up.state_dict())
    return up, mixed


def test_G1_equivalence_with_upstream():
    # No interventional context rows: K = 0, so ctx_int is all False.
    cfg = grouped_cfg(p_obs=1.0, k_max=0, qry_groups=[1])
    batch = grouped_batch(G=1, cfg=cfg)
    assert not batch["ctx_int"].any()
    out = prep(batch)
    up, mixed = _upstream_and_mixed()
    with torch.no_grad():
        a = mixed(
            context_data=out["context"],
            target_data=out["target"],
            intervention_index=out["intervention_index"],
            outcome_index=out["outcome_index"],
            context_int_mask=out["context_int_mask"],
        )
        b = up(
            context_data=out["context"],
            target_data=out["target"],
            intervention_index=out["intervention_index"],
            outcome_index=out["outcome_index"],
        )
    for field in ("pred_mean", "pred_std", "weights"):
        assert torch.allclose(getattr(a, field), getattr(b, field), atol=1e-6)


def test_group_isolation():
    _, model = _upstream_and_mixed()
    with torch.no_grad():
        for p in model.encoder.context_intervened.parameters():
            p.normal_(0, 0.5)
    batch = grouped_batch(G=2)
    G = 2

    def run(bt):
        out = prep(bt)
        with torch.no_grad():
            res = model(
                context_data=out["context"],
                target_data=out["target"],
                intervention_index=out["intervention_index"],
                outcome_index=out["outcome_index"],
                context_int_mask=out["context_int_mask"],
            )
        return torch.cat([res.pred_mean, res.pred_std, res.weights], -1)

    before = run(batch)
    nq_g = batch["qry_x"].shape[1] // G
    batch["qry_x"][:, nq_g:] += 5.0  # perturb group 1's query rows
    after = run(batch)
    assert torch.allclose(
        before[0::G], after[0::G], atol=1e-6
    )  # group 0 unchanged
    assert not torch.allclose(before[1::G], after[1::G], atol=1e-4)


def test_trainer_runs_train_validate_test(tmp_path):
    cfg = grouped_cfg()
    train = MixedRegimeTaskDataset(
        cfg, 4, base_seed=0, split="train", iterations_per_epoch=3
    )
    val = MixedRegimeTaskDataset(
        cfg,
        4,
        base_seed=0,
        split="val",
        iterations_per_epoch=2,
        fresh_each_epoch=False,
    )
    test = MixedRegimeTaskDataset(
        cfg,
        4,
        base_seed=0,
        split="test",
        iterations_per_epoch=2,
        fresh_each_epoch=False,
    )
    model = MixedContextMoGModel(tiny_model_config())
    dataconfig, optimconfig, trainingconfig, loggingconfig = (
        convert_trainerdict_to_config(
            {
                "batch_size": 4,
                "num_workers": 0,
                "cntxt_split": [0.05, 0.75],
                "sample_size": 0,
                "train_dtype": "float32",
                "eval_dtype": "float32",
                "pin_memory": False,
                "normalise": True,
                "optimizer": torch.optim.Adam(model.parameters(), lr=1e-3),
                "learning_rate": 1e-3,
                "lr_warmup_ratio": 0.0,
                "epochs": 2,
                "gradient_clip_val": 1.0,
                "device": "cpu",
                "save_dir": Path(tmp_path),
                "use_wandb": False,
                "log_step": 1000,
                "save_checkpoint_every_n_steps": 1000,
                "plot_validation_samples": False,
                "num_validation_plots": 0,
            }
        )
    )
    trainer = MixedContextTrainer(
        train_dataset=train,
        validation_dataset=val,
        test_dataset=test,
        model=model,
        dataconfig=dataconfig,
        optimizerconfig=optimconfig,
        trainingconfig=trainingconfig,
        loggingconfig=loggingconfig,
        callbacks=[],
        seed=0,
    )
    assert trainer.steps_per_epoch == 3
    trainer.train()
    assert trainer.global_step == 6
    v1, v2 = trainer.validate()["val_loss"], trainer.validate()["val_loss"]
    assert np.isfinite(v1) and v1 == v2  # validation is deterministic
    metrics = trainer.test()
    assert np.isfinite(metrics["test_loss_all_mean"])
    assert any(k.startswith("test_loss_group_type=") for k in metrics)
    assert any(k.startswith("test_loss_K=") for k in metrics)
    assert any(k.startswith("test_loss_G=") for k in metrics)
