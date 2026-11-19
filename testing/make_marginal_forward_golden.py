"""Record golden outputs of the five marginal mixture models BEFORE the
prediction-head refactor (each family's forward split into
`_encode_query_tokens` and `_predict_from_query_tokens`).

Run once, from the repository root, on origin/bmrc (the code before the
refactor), e.g. in a worktree:

    git worktree add ../alcfm-bmrc origin/bmrc
    cp testing/make_marginal_forward_golden.py ../alcfm-bmrc/testing/
    (cd ../alcfm-bmrc && PYTHONPATH=. python testing/make_marginal_forward_golden.py)
    cp ../alcfm-bmrc/testing/fixtures/marginal_forward_golden.pt testing/fixtures/
    git worktree remove --force ../alcfm-bmrc

Writes testing/fixtures/marginal_forward_golden.pt with, for each encoding
(r1, layerwise with the shared flag network, extra_node, reg_axis,
reg_axis_full; scale_conditioning "stats"):
- param_names: state-dict keys, in order;
- param_sums: the sum of every state-dict tensor (detects a change of
  construction, as opposed to a change of the forward pass);
- outputs: eval-mode pred_mean / pred_std / weights in float64.
Every parameter is moved by noise seeded by its name, so the zero-initialised
flag, injection and scale layers act on the outputs. The inputs (stored too)
have a mixed context: observational rows plus two interventional regimes on
different variables. No state dict is stored (file size); the test rebuilds
the models from the same seed.

Uses only code and constructor arguments that exist both before and after
the refactor. CPU, float64, deterministic.
"""

import zlib
from pathlib import Path

import torch

from CITNP.models.mixed_context_model import INTERVENTION_ENCODINGS
from CITNP.utils.configs import CausalInfModelConfig

OUT = Path(__file__).resolve().parent / "fixtures" / "marginal_forward_golden.pt"
ENCODINGS = ("r1", "layerwise", "extra_node", "reg_axis", "reg_axis_full")
NUM_NODES = 4


def model_config() -> CausalInfModelConfig:
    """Every field explicit: the defaults differ between the two branches."""
    return CausalInfModelConfig(
        d_model=32,
        emb_depth=2,
        dim_feedforward=64,
        nhead=4,
        dropout=0.0,
        num_layers_encoder=4,
        num_nodes=NUM_NODES,
        sample_attn_mode="MHCA",
        linear_attention=False,
        mean_loss_across_samples=False,
        device="cpu",
        dtype="float32",
        num_mixture_components=3,
        scale_conditioning="stats",
        reg_axis_num_blocks=None,
        reg_axis_num_memory_tokens=4,
        reg_axis_target_flag="injected",
        reg_axis_count_feature=True,
        share_intervention_encoder=True,  # read by layerwise only
    )


def build(encoding: str) -> torch.nn.Module:
    """Seeded model of `encoding`, every parameter perturbed, float64, eval."""
    torch.manual_seed(0)
    model = INTERVENTION_ENCODINGS[encoding](model_config()).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def golden_inputs() -> dict:
    """B = 3 tasks, 18 context rows (8 observational, 5 + 5 rows of two
    single-target regimes on different variables, shuffled), 5 query rows."""
    g = torch.Generator().manual_seed(1)
    B, Nc, Nq, D = 3, 18, 5, NUM_NODES
    mask = torch.zeros(B, Nc, D, dtype=torch.bool)
    for b, (first, second) in enumerate([(0, 2), (1, 3), (3, 0)]):
        mask[b, 8:13, first] = True
        mask[b, 13:18, second] = True
        mask[b] = mask[b, torch.randperm(Nc, generator=g)]
    scale = torch.stack(
        [0.3 * torch.randn(B, D, generator=g), torch.randn(B, D, generator=g)],
        dim=-1,
    )
    return {
        "context_data": torch.randn(B, Nc, D, 1, generator=g).double(),
        "target_data": torch.randn(B, Nq, D, 1, generator=g).double(),
        "intervention_index": torch.tensor([0, 1, 3]),
        "outcome_index": torch.tensor([2, 3, 1]),
        "context_int_mask": mask,
        "context_scale": scale.double(),
    }


def record(model: torch.nn.Module, inputs: dict) -> dict:
    with torch.no_grad():
        out = model(**inputs)
    return {
        "param_names": list(model.state_dict()),
        "param_sums": {
            k: v.double().sum().item() for k, v in model.state_dict().items()
        },
        "outputs": {
            "pred_mean": out.pred_mean,
            "pred_std": out.pred_std,
            "weights": out.weights,
        },
    }


def main() -> None:
    inputs = golden_inputs()
    fixture = {
        "inputs": inputs,
        "cases": {enc: record(build(enc), inputs) for enc in ENCODINGS},
    }
    OUT.parent.mkdir(exist_ok=True)
    torch.save(fixture, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
