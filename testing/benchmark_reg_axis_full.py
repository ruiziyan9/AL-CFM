"""Cost of one training step of reg_axis_full on a worst-case batch, next to
layerwise and reg_axis on the same batch. Not a pytest test.

    python testing/benchmark_reg_axis_full.py \
        --model_config CITNP/experiments/configs/oracle_d3_model_m_reg_axis_full_stats.yaml \
        --batch_size 256 --num_context 256 --num_regimes 4 --device cuda

The batch is synthetic, with the regime split that is worst for the packed
keys of reg_axis_full: in every task one regime holds Nc - R + 1 context rows
and the other R - 1 regimes one row each, so N = Nc - R + 1 and the packed
keys have R * N positions for Nc real rows. Values are standard normal.

For each encoding the script builds the model, runs one warm-up step and
`--steps` forward + backward steps in train mode under the training dtype
(model and batch cast to `--dtype`, as MixedContextTrainer does), and prints
the median step time and the peak memory: `torch.cuda.max_memory_allocated`
on GPU, the peak resident set size of the process on CPU. Every encoding
runs in its own subprocess, so the peaks do not mix.

The encodings are reg_axis_full, layerwise and reg_axis (plus the encoding of
`--model_config`, if it is another one), all with the size keys of the config
(d_model, nhead, num_layers_encoder, ...); choose others with `--encodings`.

Notation: B = batch_size * question_groups (the flattened batch the model
sees), Nc context rows, Nq query rows, D variables, R regimes, N rows of the
largest regime.
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time

import torch
import yaml  # type: ignore[import-untyped]

from CITNP.models.mixed_context_model import (
    INTERVENTION_ENCODINGS,
    resolve_model_class,
)
from CITNP.models.regime_axis import REG_AXIS_CONFIG_KEYS, group_regimes
from CITNP.models.regime_axis_full import REG_AXIS_FULL_CONFIG_KEYS
from CITNP.utils.configs import CausalInfModelConfig

DEFAULT_ENCODINGS = ("reg_axis_full", "layerwise", "reg_axis")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_config", required=True, help="model YAML")
    parser.add_argument("--batch_size", type=int, required=True, help="tasks")
    parser.add_argument(
        "--num_context", type=int, required=True, help="context rows Nc"
    )
    parser.add_argument(
        "--num_regimes", type=int, required=True, help="regimes R per task"
    )
    parser.add_argument("--device", required=True, help="cpu or cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", help="training dtype (mixed_train --dtype)"
    )
    parser.add_argument(
        "--question_groups",
        type=int,
        default=1,
        help="question groups G per task; the model sees batch_size * G elements",
    )
    parser.add_argument(
        "--num_query", type=int, default=48, help="query rows Nq per group"
    )
    parser.add_argument("--num_variables", type=int, default=3, help="D")
    parser.add_argument("--steps", type=int, default=5, help="timed steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--encodings",
        default=None,
        help=f"comma-separated encodings to run (default: "
        f"{', '.join(DEFAULT_ENCODINGS)}, plus the config's own)",
    )
    parser.add_argument(
        "--single",
        default=None,
        help=argparse.SUPPRESS,  # one encoding, JSON out
    )
    return parser.parse_args(argv)


def model_config_for(
    settings: dict, encoding: str, num_variables: int, device: str
) -> CausalInfModelConfig:
    """Model config with the size keys of the YAML `settings` for `encoding`.

    Encoding-specific keys are kept only where they apply: the reg_axis_*
    keys for the regime-axis encodings (without the memory-token key for
    reg_axis_full) and share_intervention_encoder for layerwise.
    """
    allowed: tuple[str, ...] = ()
    if encoding == "reg_axis":
        allowed = REG_AXIS_CONFIG_KEYS
    elif encoding == "reg_axis_full":
        allowed = REG_AXIS_FULL_CONFIG_KEYS
    return CausalInfModelConfig(
        d_model=settings["d_model"],
        emb_depth=settings["emb_depth"],
        dim_feedforward=settings["dim_feedforward"],
        nhead=settings["nhead"],
        dropout=0.0,
        num_layers_encoder=settings["num_layers_encoder"],
        num_nodes=num_variables,
        sample_attn_mode=settings["sample_attn_mode"],
        linear_attention=settings.get("linear_attention", False),
        mean_loss_across_samples=False,
        device=device,
        dtype="float32",
        num_mixture_components=settings["num_mixture_components"],
        scale_conditioning=settings.get("scale_conditioning", "none"),
        **{k: settings[k] for k in allowed if k in settings},
        share_intervention_encoder=bool(
            encoding == "layerwise"
            and settings.get("share_intervention_encoder", False)
        ),
    )


def check_num_regimes(num_regimes: int, num_context: int, num_variables: int):
    """R distinct target sets need R <= 2**D, and one row each needs R <= Nc."""
    limit = min(num_context, 2**num_variables)
    if not 1 <= num_regimes <= limit:
        raise ValueError(
            f"num_regimes={num_regimes} must be in [1, min(num_context, "
            f"2**num_variables)] = [1, {limit}]"
        )


def worst_case_batch(
    batch_size: int,
    num_context: int,
    num_regimes: int,
    num_query: int,
    num_variables: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Synthetic model inputs with the worst-case regime split.

    Regime r has the target set whose bitmask is r (regime 0 is
    observational), so R <= 2**D. Regime 0 gets Nc - R + 1 rows and every
    other regime one row; the rows of each task are shuffled.

    Returns:
        context_data [B, Nc, D, 1], target_data [B, Nq, D, 1],
        intervention_index / outcome_index [B] long, context_int_mask
        [B, Nc, D] bool, context_scale [B, D, 2] (float32, CPU).
    """
    B, Nc, R, D = batch_size, num_context, num_regimes, num_variables
    check_num_regimes(R, Nc, D)
    g = torch.Generator().manual_seed(seed)
    # regime of each row: R - 1 single rows, the rest in regime 0
    regime = torch.zeros(B, Nc, dtype=torch.long)
    regime[:, : R - 1] = torch.arange(1, R)
    shuffle = torch.rand(B, Nc, generator=g).argsort(dim=1)
    regime = regime.gather(1, shuffle)
    bits = 2 ** torch.arange(D)
    mask = (regime[..., None] & bits) > 0  # [B, Nc, D]
    treatment = torch.arange(B) % D
    return dict(
        context_data=torch.randn(B, Nc, D, 1, generator=g),
        target_data=torch.randn(B, num_query, D, 1, generator=g),
        intervention_index=treatment,
        outcome_index=(treatment + 1) % D,
        context_int_mask=mask,
        context_scale=torch.randn(B, D, 2, generator=g),
    )


def run_single(args: argparse.Namespace, encoding: str) -> dict:
    """Build the model of `encoding`, time the steps and return the result."""
    with open(args.model_config) as f:
        settings = yaml.safe_load(f)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    model_cls, _ = resolve_model_class(encoding)
    model = model_cls(
        model_config_for(settings, encoding, args.num_variables, args.device)
    )
    num_parameters = sum(p.numel() for p in model.parameters())
    model = model.to(device).to(dtype).train()

    batch = worst_case_batch(
        args.batch_size * args.question_groups,
        args.num_context,
        args.num_regimes,
        args.num_query,
        args.num_variables,
        args.seed,
    )
    grouping = group_regimes(batch["context_int_mask"])
    batch = {
        k: v.to(device, dtype) if v.is_floating_point() else v.to(device)
        for k, v in batch.items()
    }
    on_gpu = device.type == "cuda"

    def step() -> None:
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = model.calculate_loss(
            model_output=output,
            target=batch["target_data"],
            outcome_index=batch["outcome_index"],
        )
        loss.backward()
        if on_gpu:
            torch.cuda.synchronize(device)

    step()  # warm-up (allocator, kernels)
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(args.steps):
        start = time.perf_counter()
        step()
        times.append(time.perf_counter() - start)
    if on_gpu:
        peak_bytes = torch.cuda.max_memory_allocated(device)
    else:  # ru_maxrss is in kilobytes on Linux; includes the warm-up step
        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "encoding": encoding,
        "parameters": num_parameters,
        "median_step_seconds": statistics.median(times),
        "min_step_seconds": min(times),
        "peak_memory_gib": peak_bytes / 2**30,
        "R": grouping.num_slots,
        "N": int(grouping.n_r.max()),
    }


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    if args.single is not None:
        print(json.dumps(run_single(args, args.single)))
        return

    with open(args.model_config) as f:
        settings = yaml.safe_load(f)
    _, own = resolve_model_class(
        settings.get("intervention_encoding"), settings.get("model_class")
    )
    if args.encodings is None:
        encodings = list(DEFAULT_ENCODINGS)
        if own in INTERVENTION_ENCODINGS and own not in encodings:
            encodings.append(own)
    else:
        encodings = [e.strip() for e in args.encodings.split(",")]
    unknown = [e for e in encodings if e not in INTERVENTION_ENCODINGS]
    if unknown:
        raise SystemExit(
            f"unknown encodings {unknown}; choose from "
            f"{sorted(INTERVENTION_ENCODINGS)}"
        )

    B = args.batch_size * args.question_groups
    Nc, R = args.num_context, args.num_regimes
    try:
        check_num_regimes(R, Nc, args.num_variables)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    N = Nc - R + 1
    memory = (
        "torch.cuda.max_memory_allocated"
        if torch.device(args.device).type == "cuda"
        else "peak resident set size of the process"
    )
    print(f"model config: {args.model_config}")
    print(
        f"batch: B = {args.batch_size} tasks x {args.question_groups} groups "
        f"= {B}, Nc = {Nc}, Nq = {args.num_query}, D = {args.num_variables}, "
        f"R = {R}, worst-case split: N = {N} (R * N / Nc = {R * N / Nc:.2f})"
    )
    print(
        f"device {args.device}, dtype {args.dtype}, {args.steps} forward + "
        f"backward steps after one warm-up; memory = {memory}"
    )
    header = f"{'encoding':<15}{'parameters':>12}{'median step':>14}{'min step':>12}{'peak memory':>14}"
    print(header)
    for encoding in encodings:
        done = subprocess.run(
            [sys.executable, __file__, *argv, "--single", encoding],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            stderr_lines = done.stderr.strip().splitlines()
            print(
                f"{encoding:<15}failed: {stderr_lines[-1] if stderr_lines else 'no output'}"
            )
            continue
        result = json.loads(done.stdout.strip().splitlines()[-1])
        assert result["R"] == R and result["N"] == N
        print(
            f"{encoding:<15}{result['parameters']:>12,}"
            f"{result['median_step_seconds']:>12.3f} s"
            f"{result['min_step_seconds']:>10.3f} s"
            f"{result['peak_memory_gib']:>10.2f} GiB"
        )


if __name__ == "__main__":
    main()
