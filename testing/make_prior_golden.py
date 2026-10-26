"""Record golden outputs of the task generator and the oracle BEFORE the
prior refactor (steps S1 onwards).

Run once, from the repository root, on the commit that still has the original
`LinearNonIdentifiable` generator and oracle (the S0.2 commit):

    PYTHONPATH=. python testing/make_prior_golden.py

Writes to testing/golden/prior/:

    generator_<name>.npz   20 tasks per setting (oracle_d3, oracle_d3_strong_v9,
                           signed_mean), every array field
    oracle.npz             log marginal likelihood of all 25 DAGs, 3 datasets
                           x 3 (eta, weight_mean) settings

Runs on CPU; deterministic, so rerunning reproduces identical files. Never
regenerate the goldens to make `test_prior_golden.py` pass.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_golden_spec import (
    GENERATOR_SETTINGS,
    GOLDEN,
    REPO,
    generator_arrays,
    oracle_arrays,
)


def main() -> None:
    os.chdir(REPO)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name in GENERATOR_SETTINGS:
        path = GOLDEN / f"generator_{name}.npz"
        np.savez(path, **generator_arrays(name))
        print("wrote", path.relative_to(REPO))
    path = GOLDEN / "oracle.npz"
    np.savez(path, **oracle_arrays())
    print("wrote", path.relative_to(REPO))


if __name__ == "__main__":
    main()
