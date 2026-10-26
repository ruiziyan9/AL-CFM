"""
How often does the generator reject a task and resample it?

    python -m CITNP.prior_diagnostics resample-rate \
        --data_config CITNP/experiments/configs/oracle_d3_iw_a15.yaml \
        --n_tasks 20000

A task is regenerated when its data are non-finite or exceed `max_abs_value`
(see generate_task). The accepted tasks are then a truncation of the prior, so
the oracle (which assumes the untruncated prior) is only exact if this is rare.
Reported: rejected attempts / all attempts (aim: < 0.1%), split by reason.
"""

from __future__ import annotations

import argparse

from tqdm import trange

from CITNP.datasets import mixed_regime_generator as mrg
from CITNP.datasets.create_mixed_regime_chunks import load_config

__all__ = ["resample_rate"]


def resample_rate(
    cfg,
    n_tasks: int,
    base_seed: int = 0,
    num_variables: int | None = None,
    progress: bool = False,
) -> dict:
    """Generate `n_tasks` tasks and count rejected attempts.

    Args:
    ----------
    cfg : MixedRegimeConfig
        Prior to test.

    n_tasks : int
        Number of tasks (each may take several attempts).

    base_seed : int
        Tasks use make_key(SOURCE_ADHOC, 0, base_seed, i).

    num_variables : int | None
        Fixes D (default: drawn from the config).

    Returns:
    ----------
    dict with n_tasks, n_attempts, n_rejected, n_nonfinite, n_too_large,
    n_failed (tasks still rejected after max_resample_attempts) and
    rate = n_rejected / n_attempts.
    """
    before = dict(mrg.RESAMPLE_COUNTS)
    n_failed = 0
    it = trange(n_tasks) if progress else range(n_tasks)
    for i in it:
        try:
            mrg.generate_task(
                cfg,
                mrg.make_key(mrg.SOURCE_ADHOC, 0, base_seed, i),
                num_variables=num_variables,
            )
        except RuntimeError:
            n_failed += 1
    nonfinite = mrg.RESAMPLE_COUNTS["nonfinite"] - before["nonfinite"]
    too_large = mrg.RESAMPLE_COUNTS["too_large"] - before["too_large"]
    rejected = nonfinite + too_large
    attempts = (
        n_tasks + rejected - n_failed
    )  # each failed task ends on a rejection
    return {
        "n_tasks": n_tasks,
        "n_attempts": attempts,
        "n_rejected": rejected,
        "n_nonfinite": nonfinite,
        "n_too_large": too_large,
        "n_failed": n_failed,
        "rate": rejected / max(attempts, 1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--n_tasks", type=int, default=20000)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--num_variables", type=int, default=None,
                        help="fix D (default: D drawn from the config)")  # fmt: skip
    args = parser.parse_args(argv)
    cfg = load_config(args.data_config)
    r = resample_rate(cfg, args.n_tasks, args.base_seed, args.num_variables, True)
    for k, v in r.items():
        print(f"{k:12s} {v:.3e}" if k == "rate" else f"{k:12s} {v}")
