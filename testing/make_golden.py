"""Record golden outputs of the evaluation code BEFORE the refactor.

Run once, from the repository root, on the commit that still has the original
`CITNP/evaluation/` (the commit that added this script):

    PYTHONPATH=. python testing/make_golden.py [--ckpt TRAINED.pt]

Writes to testing/golden/ (tiny random model, always) and, when --ckpt or
$ALCFM_TRAINED_CKPT is given, to testing/golden/trained/ (size-m config):

    tiny_model.pt                 state dict, size-s config, torch.manual_seed(0)
    bank.pkl                      oracle_bank, per_bucket=2 (model-independent)
    kshot.csv, kshot_by_t.csv     eval_d3_kshot
    suite.csv                     eval_d3_oracle
    bank_gap.json                 OracleGapCallback._evaluate on bank.pkl
    wrappers.npz                  model_mixtures / model_logpdf / mog_predictive

Runs on CPU; deterministic, so rerunning reproduces identical files.
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import (  # noqa: E402
    BANK_ARGS,
    DATA_CFG,
    GOLDEN,
    KSHOT_ARGS,
    REPO,
    SUITE_ARGS,
    build_tiny_model,
    callback_bank_gap,
    model_sets,
    wrapper_outputs,
)


def run_main(main, args: list[str]) -> None:
    argv = sys.argv
    sys.argv = ["prog", *args]
    try:
        main()
    finally:
        sys.argv = argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="trained checkpoint (size-m config)")
    a = ap.parse_args()

    # These names exist only on the pre-refactor baseline (commit abbb68c),
    # which is the only code this script is meant to run on.
    from CITNP.evaluation import eval_d3_kshot, eval_d3_oracle, oracle_bank
    from CITNP.evaluation.eval_d3_kshot import (  # type: ignore[attr-defined]
        model_mixtures,
    )
    from CITNP.evaluation.eval_d3_oracle import (  # type: ignore[attr-defined]
        load_model,
        model_logpdf,
    )
    from CITNP.trainer.monitor_callbacks import (  # type: ignore[attr-defined]
        mog_predictive,
    )

    os.chdir(REPO)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    torch.save(build_tiny_model().state_dict(), GOLDEN / "tiny_model.pt")

    bank = GOLDEN / "bank.pkl"
    run_main(
        oracle_bank.main,
        ["--data_config", DATA_CFG, "--out", str(bank.relative_to(REPO)), *BANK_ARGS],
    )
    with open(bank, "rb") as f:
        items = pickle.load(f)["items"]

    for name, out_dir, ckpt, model_cfg in model_sets(a.ckpt):
        print(f"\n##### {name}: {ckpt}")
        out_dir.mkdir(parents=True, exist_ok=True)
        common = ["--ckpt", ckpt, "--model_config", model_cfg, "--data_config", DATA_CFG]
        rel = out_dir.relative_to(REPO)
        run_main(eval_d3_kshot.main, [*common, *KSHOT_ARGS, "--out", f"{rel}/kshot.csv"])
        for fig in ("main", "by_query", "effect"):
            (out_dir / f"kshot_{fig}.png").unlink(missing_ok=True)
        run_main(eval_d3_oracle.main, [*common, *SUITE_ARGS, "--out", f"{rel}/suite.csv"])

        model = load_model(ckpt, model_cfg, "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            gap = {
                f"normalise={n}": callback_bank_gap(model, items, n, Path(tmp))
                for n in (True, False)
            }
        with open(out_dir / "bank_gap.json", "w") as f:
            json.dump(gap, f, indent=1)
        np.savez(
            out_dir / "wrappers.npz",
            **wrapper_outputs(model, items, model_mixtures, model_logpdf, mog_predictive),
        )
        print(f"wrote golden files for {name} to {out_dir}")


if __name__ == "__main__":
    main()
