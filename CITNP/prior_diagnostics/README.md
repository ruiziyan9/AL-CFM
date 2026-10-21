# Prior diagnostics

Visualise the mixed-regime prior DGP (data-generating process) of
`CITNP/datasets/mixed_regime_generator.py` without a trained model. Every
command takes a generator config (`MixedRegimeConfig` YAML) and samples tasks
with `SOURCE_ADHOC` keys, so task i is the same task in every command for the
same `--base_seed`.

| command | module | shows |
|---|---|---|
| `examples` | `examples.py` | per task: DAG with edge width ∝ estimated standardised coefficient (blue +, red −), pairwise scatters of context rows by regime and query rows |
| `obs-vs-int` | `obs_vs_int.py` | one SCM: p(X_y), p(X_y \| X_t≈x) and p(X_y \| do(X_t=x)) for a few x, with standardised mean gaps |
| `stats` | `population.py` | over many tasks: where training (treatment, outcome) pairs sit in the DAG (direct, downstream 2 / 3+, upstream, confounded, independent) by D, and effect sizes / do-vs-conditional gaps by relation; also writes CSVs |
| `obs-vs-do` | `obs_vs_do.py` | **Main prior check** (linear-Gaussian, exact). One column per question class (reverse, confounded effect, pure confounding, unconfounded effect, unrelated): share of training questions, exact 90% bands of p(Y \| T=x) vs p(Y \| do(T=x)), their W2 gap, the context's rows by role (obs, do(T), do(confounder), do(mediator), do(Y), do(other)), and the evidence (nats) the do(T) rows carry against the observational answer. Any D; `--num_variables 3` gives the three-variable view. Writes a CSV |
| `summary` | `summary.py` | histograms of design statistics (D, K, regime sizes, do-values, …) |

```bash
python -m CITNP.prior_diagnostics examples   --config <cfg> --num_tasks 12 --out examples.pdf --png
python -m CITNP.prior_diagnostics obs-vs-int --config <cfg> --num_tasks 4 --out obs_vs_int.pdf --png
python -m CITNP.prior_diagnostics stats      --config <cfg> --num_tasks 2000 --n_probe_tasks 300 --out prior_stats.pdf
python -m CITNP.prior_diagnostics summary    --config <cfg> --num_tasks 2000 --out prior.png
python -m CITNP.prior_diagnostics obs-vs-do  --config <cfg> --num_tasks 2000 --out obs_vs_do.pdf --png [--num_variables 3]
pytest testing/test_prior_diagnostics.py testing/test_prior_obs_vs_do.py
```

Python API: `from CITNP.prior_diagnostics import probe_task, plot_obs_vs_int, edge_strengths, ...`.
`stats.py` is numpy-only and can be used without matplotlib.

## Notes

- **Probes.** `obs-vs-int` and `stats` keep task i's graph and generate, with
  the same key, one large observational regime plus one large regime per fixed
  do-value in a single generation call, so all compared distributions come from
  one SCM. Mechanisms are not stored, so they are a fresh draw from the prior
  on that graph. This needs `Design.regime_do_values` and
  `generate_task(..., permute=False)` in the generator.
- **Edge strengths** are standardised OLS coefficients of each child on its
  true parents from observational rows: exact in expectation for linear
  families, a linear summary otherwise.
- **Device.** GP/NN mechanisms run on CUDA when visible; set
  `CUDA_VISIBLE_DEVICES=""` to match CPU-generated chunks. GP families must keep
  `n_ref` + all probe rows within `max_total_rows_gp`.
- **Do-values.** `obs-vs-int` defaults to observational quantiles of X_t
  (`--do_quantiles 0.1,0.5,0.9`); `--do_values=-2,0,2` sets them directly
  (use `=` for negative numbers).
- **Exact distributions** (`linear_gaussian.py`, used by `obs-vs-do`) read
  each node's weights, noise variance and reference standardisation through
  `generate_task(..., return_debug=True)` (generator edit 3). They apply to the
  `linearnonidentifiable` family only; tasks and chunks are unchanged.
- **Question classes** (`stats.question_class`): p(Y | T) and p(Y | do(T))
  differ generically iff there is an open back-door path from T to Y, i.e. Y
  is an ancestor of T ("reverse") or some node has directed paths to T and,
  avoiding T, to Y ("confounded effect" / "pure confounding").
