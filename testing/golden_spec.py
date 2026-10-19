"""Shared specification of the evaluation regression runs.

Used by `testing/make_golden.py` (records the golden outputs with the
pre-refactor code) and `testing/test_evaluation_regression.py` (reruns them
with the current code), so both use exactly the same arguments, models and
inputs. Imports only code that exists before and after the refactor.
"""

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "testing" / "golden"

# Relative to REPO; the bank pickle stores the data-config string verbatim,
# so always run from REPO with these exact strings.
DATA_CFG = "CITNP/experiments/configs/oracle_d3.yaml"
TINY_CFG = "CITNP/experiments/configs/oracle_d3_model_s.yaml"
TRAINED_CFG = "CITNP/experiments/configs/oracle_d3_model_m.yaml"
TRAINED_ENV = "ALCFM_TRAINED_CKPT"
D = 3

KSHOT_ARGS = [
    "--n_seeds", "2", "--k_list", "0,1,5,20", "--n_y", "16", "--n_mc", "50",
    "--device", "cpu",
]  # fmt: skip
SUITE_ARGS = [
    "--n_seeds", "2", "--n_query", "32", "--n_mc", "50", "--device", "cpu",
]  # fmt: skip
BANK_ARGS = ["--per_bucket", "2"]

# Inputs of the low-level model-wrapper comparison.
GRID = np.linspace(-3.0, 3.0, 13)
N_WRAPPER_CONTEXTS = 4
QUERIES = [(T, Y) for T in range(D) for Y in range(D) if T != Y]


def tiny_model_cfg() -> dict:
    with open(REPO / TINY_CFG) as f:
        return yaml.safe_load(f)


def build_tiny_model() -> torch.nn.Module:
    """Random-initialised size-s model, seed 0. Same config as `load_model`."""
    from CITNP.models.mixed_context_model import MixedContextMoGModel
    from CITNP.utils.configs import CausalInfModelConfig

    m = tiny_model_cfg()
    cfg = CausalInfModelConfig(
        d_model=m["d_model"],
        emb_depth=m["emb_depth"],
        dim_feedforward=m["dim_feedforward"],
        nhead=m["nhead"],
        dropout=0.0,
        num_layers_encoder=m["num_layers_encoder"],
        num_nodes=D,
        sample_attn_mode=m["sample_attn_mode"],
        linear_attention=m.get("linear_attention", False),
        mean_loss_across_samples=False,
        device="cpu",
        dtype="float32",
        num_mixture_components=m["num_mixture_components"],
    )
    torch.manual_seed(0)
    return MixedContextMoGModel(cfg)


def model_sets(trained_ckpt: str | None) -> list[tuple[str, Path, str, str]]:
    """(name, golden dir, checkpoint, model config) for every model to check."""
    out = [("tiny", GOLDEN, str(GOLDEN / "tiny_model.pt"), TINY_CFG)]
    ckpt = trained_ckpt or os.environ.get(TRAINED_ENV)
    if ckpt:
        out.append(("trained", GOLDEN / "trained", ckpt, TRAINED_CFG))
    return out


def callback_bank_gap(
    model: torch.nn.Module, items: list[dict], normalise: bool, tmp_dir: Path
) -> dict[str, float]:
    """One OracleGapCallback evaluation (step 0) of `model` on the bank items;
    returns the CSV row it writes, minus the step column."""
    import csv
    from types import SimpleNamespace

    from CITNP.trainer.monitor_callbacks import OracleGapCallback

    cb = OracleGapCallback("unused", every=1, flag_ablation=True)
    cb.items = items
    cb.path = Path(tmp_dir) / f"oracle_gap_norm{int(normalise)}.csv"
    if cb.path.exists():
        cb.path.unlink()
    trainer = SimpleNamespace(
        model=model, device="cpu", normalise=normalise, global_step=0
    )
    cb._evaluate(trainer)
    with open(cb.path) as f:
        (row,) = list(csv.DictReader(f))
    row.pop("step")
    return {k: float(v) for k, v in row.items()}


def wrapper_outputs(
    model: torch.nn.Module,
    items: list[dict],
    mixtures_fn: Callable,
    logpdf_fn: Callable,
    mog_fn: Callable,
) -> dict[str, np.ndarray]:
    """Outputs of the three model wrappers on fixed inputs.

    Signatures are those of the pre-refactor functions:
        mixtures_fn(model, ctx_x, ctx_int, grid, device, zero_flags)
            -> (mu, sd, logw) [6, n_t, K] for all six QUERIES   (eval_d3_kshot)
        logpdf_fn(model, ctx_x, ctx_int, T, Y, t, y, device, zero_flags)
            -> log density [n]                                  (eval_d3_oracle)
        mog_fn(model, item, device, normalise, zero_flags)
            -> (logp, pit, eff) [n]                             (monitor_callbacks)
    """
    out: dict[str, np.ndarray] = {}
    for c, it in enumerate(items[:N_WRAPPER_CONTEXTS]):
        x, m = it["ctx_x"], it["ctx_int"]
        for zf in (0, 1):
            mu, sd, logw = mixtures_fn(model, x, m, GRID, "cpu", bool(zf))
            for name, v in zip(("mu", "sd", "logw"), (mu, sd, logw)):
                out[f"mix_c{c}_zf{zf}_{name}"] = v.cpu().numpy()
            for T, Y in QUERIES:
                y = np.random.default_rng([c, T, Y]).normal(0.0, 1.5, len(GRID))
                out[f"lpdf_c{c}_T{T}_Y{Y}_zf{zf}"] = np.asarray(
                    logpdf_fn(model, x, m, T, Y, GRID, y, "cpu", bool(zf))
                )
    for i, it in enumerate(items):
        for norm in (1, 0):
            for zf in (0, 1):
                lp, pit, eff = mog_fn(model, it, "cpu", bool(norm), bool(zf))
                key = f"mog_i{i}_n{norm}_zf{zf}"
                out[key + "_lp"] = np.asarray(lp)
                out[key + "_pit"] = np.asarray(pit)
                out[key + "_eff"] = np.asarray(eff)
    return out
