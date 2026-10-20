"""Command-line entry point for prior diagnostics (dispatch only).

    python -m CITNP.prior_diagnostics examples   --config CFG --out examples.pdf [--png]
    python -m CITNP.prior_diagnostics obs-vs-int --config CFG --out obs_vs_int.pdf [--png]
    python -m CITNP.prior_diagnostics stats      --config CFG --out prior_stats.pdf [--png]
    python -m CITNP.prior_diagnostics summary    --config CFG --out prior.png
    python -m CITNP.prior_diagnostics obs-vs-do  --config CFG --out obs_vs_do.pdf [--png] [--num_variables 3]

`python -m CITNP.prior_diagnostics <command> -h` lists each command's options.
"""

import importlib
import sys

COMMANDS = {
    "examples": "CITNP.prior_diagnostics.examples",
    "obs-vs-int": "CITNP.prior_diagnostics.obs_vs_int",
    "stats": "CITNP.prior_diagnostics.population",
    "summary": "CITNP.prior_diagnostics.summary",
    "obs-vs-do": "CITNP.prior_diagnostics.obs_vs_do",
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    importlib.import_module(COMMANDS[argv[0]]).main(argv[1:])


if __name__ == "__main__":
    main()
