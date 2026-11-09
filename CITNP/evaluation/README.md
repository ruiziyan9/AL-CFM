# Evaluation

`python -m CITNP.evaluation <command> --help` lists every command; the
module docstring of `CITNP/evaluation/__main__.py` gives example calls.

## Flag-propagation diagnostic (`flag-propagation`)

Measures how removing context intervention flags (1 -> 0) changes the
encoder's hidden states, stage by stage, and the model's predictive
distributions. Forward passes only; the model and checkpoint are never
changed. Code: `flag_propagation.py` (computation), `plot_flag_propagation.py`
(figures); the module docstrings hold the full definitions.

Supported architectures: `layerwise` (shared or separate flag MLP) and
`extra_node`. `r1` and `reg_axis` are rejected.

```bash
# one run (writes CSVs, JSON, optional NPZ and figures into --out)
python -m CITNP.evaluation flag-propagation \
  --ckpt path/to/model.pt --model_config path/to/model.yaml \
  --bank path/to/bank.pkl --num_nodes 3 \
  --max_contexts 100 --sources_per_context 3 --regime_sources_per_context 1 \
  --ablation both --seed 0 --save_examples \
  --save_source_manifest manifest.csv --out results/flag_propagation/layerwise

# the same contexts and sources for another architecture
python -m CITNP.evaluation flag-propagation --ckpt ... --model_config ... \
  --bank path/to/bank.pkl --source_manifest manifest.csv --seed 0 \
  --out results/flag_propagation/extra_node

# figures only; several runs -> architecture comparison (C) and KL (E)
python -m CITNP.evaluation flag-propagation --plot_only \
  results/flag_propagation/layerwise results/flag_propagation/extra_node \
  --out results/flag_propagation/compare
```

Other options: `--device`, `--no_normalise`, `--no_injection_decomposition`,
`--max_batch_size` (forward items per pass, including the baseline and the
control; >= 3), `--num_examples`, `--kl_samples` (default 1000), `--float64`.

**What it does.** For each selected context, inputs and the per-task
normalisation are prepared once from the original mask; only
`context_int_mask` changes. Two ablation modes: one flag (`single_flag`), or
every flag of one variable in one regime (`regime`). On banks from
`tasks.build_bank` every interventional regime has a single target, so a
regime-level ablation relabels the whole regime as observational. For
layer-wise models three injection modes separate flag information that
entered before any token mixing (first injection = embedding +
`inject_sample[0]`) from information re-injected later (`inject_sample[i]`,
i >= 1, and every `inject_node[i]`); the `embed` stage is recorded before
`inject_sample[0]`.

**Outputs** (in `--out`):
- `flag_propagation_summary.csv`: per context / source / injection mode /
  stage / destination category: mean absolute, RMS, relative and
  stage-normalised L2 change, max, number of destinations, initial source
  magnitude, net change between stages, and the noise floor.
- `flag_propagation_predictions.csv`: per query point: predictive KL (closed
  form for one Gaussian, else Monte Carlo with its standard error), mean and
  standard-deviation shifts, and the noise-floor KL.
- `flag_propagation_sources.csv`: the source manifest (contexts, questions,
  regime structure, sources) for paired runs.
- `flag_propagation_examples.npz` (with `--save_examples`): per-token maps for
  the example heatmaps.
- `flag_propagation_metadata.json`: architecture, dtype, autocast state,
  hook location, injection grouping, regime-label source, KL method,
  tolerances, metric definitions, checkpoint hash, code commit.
- Figures A (example heatmaps), B (category curves, 95% context bootstrap),
  D (layer-wise injection decomposition), E (predictive KL), F (net
  sensitivity change); with several runs, C (architecture comparison).

**Noise floor.** Every forward chunk ends with an identical-mask control
item. Items of one batch are not bitwise identical (CPU/GPU kernels round
differently by batch position), so the control's difference from the
baseline, aggregated with each source's own categories, is reported as
`noise_floor_*` and drawn as a grey band. It is about 1e-14 of the hidden
norms in float64 and about 1e-7 in float32. **Use `--float64` for mechanism
validation** (Stage A), where exact zeros are meaningful; float32 is enough
for aggregate comparisons. Automatic mixed precision is always disabled for
this command.

**Paired comparisons.** Compare architectures on the same bank with the same
`--source_manifest` (validated against the bank: context fingerprint,
question, flagged source cells, regime membership) and the same `--seed`,
which also fixes the Monte Carlo KL draws. Use `stage_normalised_l2` across
architectures, as raw hidden-state scales differ.

**Reading the results.** Sensitivity is computational dependence of the
forward pass on the flags, not information content, intervention
understanding or causal correctness, and it does not identify the head or
path responsible. Larger is not better; judge architectures by paired
truth-KL / oracle-KL, with the predictive KL here as the bridge. A small
single-flag effect with a large regime-level effect is consistent with
sensible pooling. Removing flags is an intervention on model inputs, so the
ablated mask may be inconsistent with the unchanged values. Relative and
source-normalised metrics are NaN where their denominators are near zero.
