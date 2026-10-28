import argparse
import csv
import pickle
from pathlib import Path

import torch

from CITNP.evaluation.experiments import run_bank_gap
from CITNP.evaluation.model_api import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--no-normalise",
        action="store_true",
        help="Use only if the model was trained with --no-normalise.",
    )
    args = p.parse_args()

    device = args.device

    print(f"Loading model: {args.ckpt}")
    model = load_model(
        args.ckpt,
        args.model_config,
        device=device,
        num_nodes=3,
    )

    print(f"Loading bank: {args.bank}")
    with open(args.bank, "rb") as f:
        obj = pickle.load(f)

    bank = obj["items"] if isinstance(obj, dict) else obj

    print(f"Bank questions: {len(bank)}")
    print(f"Model-input normalisation: {not args.no_normalise}")

    with torch.no_grad():
        result = run_bank_gap(
            model,
            bank,
            device=device,
            normalise=not args.no_normalise,
            flag_ablation=False,
        )

    print("\n=== Model excess NLL over Bayes oracle ===")
    for key, value in sorted(result.items()):
        print(f"{key:35s} {value:.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()