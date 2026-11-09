"""Tiny random models and synthetic contexts shared by the flag-propagation
tests and `make_model_api_states_golden.py`.

Models are built through the repository's own config path
(`model_api.load_model`: YAML -> CausalInfModelConfig -> resolve_model_class)
from a seeded random state dict, so they are real encoders, only small and
untrained. Every parameter is moved by seeded noise, so the zero-initialised
flag and injection layers also affect the forward pass.
"""

import zlib
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

D = 3

# (name, YAML keys that select the architecture)
ARCHITECTURES = {
    "r1": {"intervention_encoding": "r1"},
    "layerwise_separate": {
        "intervention_encoding": "layerwise",
        "share_intervention_encoder": False,
    },
    "layerwise_shared": {
        "intervention_encoding": "layerwise",
        "share_intervention_encoder": True,
    },
    "extra_node": {"intervention_encoding": "extra_node"},
}

TINY_MODEL = {
    "d_model": 16,
    "emb_depth": 2,
    "dim_feedforward": 32,
    "nhead": 2,
    "num_layers_encoder": 4,
    "num_mixture_components": 3,
}


# Smaller still, for the golden fixture (keeps the .pt file small).
GOLDEN_MODEL = {"d_model": 8, "dim_feedforward": 16, "nhead": 2}


def parameter_checksum(model: torch.nn.Module) -> torch.Tensor:
    """float64 [num_parameters, 2]: sum and sum of squares of every
    parameter tensor, in state-dict order."""
    return torch.tensor(
        [[float(p.double().sum()), float(p.double().square().sum())]
         for p in model.state_dict().values()],
        dtype=torch.float64,
    )  # fmt: skip


def write_model_yaml(
    path: Path, architecture: str, *, mode: str = "MHCA", **overrides
) -> str:
    m = {**TINY_MODEL, "sample_attn_mode": mode, **ARCHITECTURES[architecture]}
    m.update(overrides)
    Path(path).write_text(yaml.safe_dump(m))
    return str(path)


def perturb(model: torch.nn.Module, scale: float = 0.3) -> torch.nn.Module:
    """Add noise seeded by parameter name to every parameter (in place)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            p.add_(scale * torch.randn(p.shape, generator=g))
    return model


def write_checkpoint(
    tmp_dir: Path,
    architecture: str,
    *,
    mode: str = "MHCA",
    num_nodes: int = D,
    seed: int = 0,
    **overrides,
) -> tuple[str, str]:
    """Write a YAML model config and a seeded random checkpoint of a tiny
    `architecture` model into `tmp_dir`; returns (checkpoint, config)."""
    from CITNP.models.mixed_context_model import resolve_model_class
    from CITNP.utils.configs import CausalInfModelConfig

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = write_model_yaml(
        tmp_dir / f"{architecture}_{mode}.yaml", architecture, mode=mode,
        **overrides,
    )  # fmt: skip
    m = yaml.safe_load(Path(cfg_path).read_text())
    cfg = CausalInfModelConfig(
        d_model=m["d_model"], emb_depth=m["emb_depth"],
        dim_feedforward=m["dim_feedforward"], nhead=m["nhead"], dropout=0.0,
        num_layers_encoder=m["num_layers_encoder"], num_nodes=num_nodes,
        sample_attn_mode=m["sample_attn_mode"], linear_attention=False,
        mean_loss_across_samples=False,
        device="cpu", dtype="float32",
        num_mixture_components=m["num_mixture_components"],
        scale_conditioning=m.get("scale_conditioning", "none"),
        share_intervention_encoder=bool(
            m.get("share_intervention_encoder", False)
        ),
    )  # fmt: skip
    cls, _ = resolve_model_class(m.get("intervention_encoding"))
    torch.manual_seed(seed)
    ckpt = tmp_dir / f"{architecture}_{mode}_{seed}.pt"
    torch.save({"model_state_dict": perturb(cls(cfg)).state_dict()}, ckpt)
    return str(ckpt), cfg_path


def build_model(
    tmp_dir: Path,
    architecture: str,
    *,
    mode: str = "MHCA",
    num_nodes: int = D,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    **overrides,
) -> torch.nn.Module:
    """Tiny random model of `architecture`, built by `load_model` from a YAML
    and a saved seeded checkpoint; returned in eval mode, CPU, `dtype`."""
    from CITNP.evaluation.model_api import load_model

    ckpt, cfg_path = write_checkpoint(
        tmp_dir, architecture, mode=mode, num_nodes=num_nodes, seed=seed,
        **overrides,
    )  # fmt: skip
    model = load_model(ckpt, cfg_path, "cpu", num_nodes=num_nodes)
    return model.to(dtype).eval()


def regime_context(seed: int = 0, n_obs: int = 10, sizes=(2, 2), targets=(0, 2)):
    """Synthetic context: an observational regime of `n_obs` rows, then one
    single-target interventional regime per (size, target); rows are NOT
    shuffled. Returns (ctx_x [N, D] float32, ctx_int [N, D] bool,
    ctx_regime [N] int)."""
    rng = np.random.default_rng(seed)
    n = n_obs + sum(sizes)
    x = rng.normal(size=(n, D)).astype(np.float32)
    ci = np.zeros((n, D), dtype=bool)
    regime = np.zeros(n, dtype=np.int64)
    row = n_obs
    for g, (size, k) in enumerate(zip(sizes, targets), start=1):
        ci[row : row + size, k] = True
        regime[row : row + size] = g
        row += size
    return x, ci, regime


def bank_items(n_contexts: int = 3, seed: int = 0) -> list[dict]:
    """Bank-like items (ctx_x, ctx_int, T, Y, t), shuffled rows, distinct
    contexts, observational + two interventional regimes each."""
    rng = np.random.default_rng(seed)
    items = []
    for c in range(n_contexts):
        targets = tuple(int(k) for k in rng.choice(D, 2, replace=False))
        x, ci, _ = regime_context(seed + c, n_obs=10, sizes=(3, 2), targets=targets)
        perm = rng.permutation(len(x))
        T, Y = (int(v) for v in rng.choice(D, 2, replace=False))
        items.append(
            dict(ctx_x=x[perm], ctx_int=ci[perm], T=T, Y=Y,
                 t=np.linspace(-1.0, 1.0, 3).astype(np.float32))
        )  # fmt: skip
    return items


def write_bank(path: Path, n_contexts: int = 4, seed: int = 0) -> str:
    """Pickle `bank_items` in the format of `python -m CITNP.evaluation bank`."""
    import pickle

    with open(path, "wb") as f:
        pickle.dump({"items": bank_items(n_contexts, seed)}, f)
    return str(path)
