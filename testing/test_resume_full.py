"""--resume_full continues a run from an epoch-boundary checkpoint."""

from pathlib import Path

import pytest
import torch

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.experiments.mixed_train import build_trainer, parse_args

CONFIGS = Path(__file__).resolve().parents[1] / "CITNP" / "experiments" / "configs"


def _args(tmp_path, epochs, *extra):
    return parse_args(
        [
            "--data_config", str(CONFIGS / "smoke_d3.yaml"),
            "--model_config", str(CONFIGS / "smoke_d3_model_extra_node_stats.yaml"),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", "4", "--iterations_per_epoch", "3",
            "--epochs", str(epochs), "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--save_checkpoint_every_n_steps", "1000",
            "--results_path", str(tmp_path), "--run_name", "r", *extra,
        ]
    )  # fmt: skip


def test_resume_full_restores_state_and_continues(tmp_path):
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    trainer, _ = build_trainer(_args(tmp_path, 1), cfg)
    trainer.train()
    ckpt = tmp_path / "results" / "r" / "last_model.pt"
    saved = torch.load(ckpt, map_location="cpu")
    assert saved["global_step"] == 3 and "best_val_loss" in saved

    # a fresh launch of the same run name is still refused
    with pytest.raises(FileExistsError):
        build_trainer(_args(tmp_path, 2), cfg)

    t2, _ = build_trainer(_args(tmp_path, 2, "--resume_full", str(ckpt)), cfg)
    assert t2.global_step == 3 and t2.current_epoch == 1
    for k, v in t2.model.state_dict().items():
        assert torch.equal(v.cpu(), saved["model_state_dict"][k]), k
    assert t2.optimizer.state_dict()["state"], "Adam state not restored"
    t2.train()
    assert t2.global_step == 6
    assert torch.load(ckpt, map_location="cpu")["global_step"] == 6


def test_resume_full_rejects_mid_epoch_checkpoint(tmp_path):
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    trainer, _ = build_trainer(_args(tmp_path, 1), cfg)
    trainer.train()
    ckpt = tmp_path / "results" / "r" / "last_model.pt"
    state = torch.load(ckpt, map_location="cpu")
    state["global_step"] = 2
    torch.save(state, ckpt)
    with pytest.raises(ValueError, match="mid-epoch"):
        build_trainer(_args(tmp_path, 2, "--resume_full", str(ckpt)), cfg)


def test_resume_full_and_resume_from_exclusive(tmp_path):
    cfg = load_config(CONFIGS / "smoke_d3.yaml")
    with pytest.raises(ValueError, match="exclusive"):
        build_trainer(
            _args(tmp_path, 1, "--resume_full", "x.pt", "--resume_from", "y.pt"), cfg
        )
