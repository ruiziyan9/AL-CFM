# Intervention-encoding diagnostics

These are three diagnostics for a trained checkpoint that test whether the
model keeps track of which context rows are interventional. They were written
to answer one question about the `extra_node` encoding
([`extra_node_encoder.py`](../models/extra_node_encoder.py)). Every row has
one metadata column:

```text
observational row             e_INT
row intervened on set S       e_INT + sum_{j in S} e_j
data cell (i, j)              f_C,r_j(x_ij) + e_j      (e_j shared with the metadata column)
```

Does adding e_INT to every row make intervened rows hard to tell apart from
observational ones inside the network? The diagnostics need no retraining.

| diagnostic | command | encodings | needs data | module |
| --- | --- | --- | --- | --- |
| 1. embedding geometry | `embedding-geometry` | `extra_node` only | no | [`embedding_geometry.py`](embedding_geometry.py) |
| 2. per-layer linear probes | `probe-intervention` | all | tasks from the prior | [`intervention_diagnostics.py`](intervention_diagnostics.py) |
| 3. label sensitivity under mask edits | `context-edits` | all | tasks from a linear-Gaussian prior (exact oracle) | [`intervention_diagnostics.py`](intervention_diagnostics.py) |

Diagnostics 2 and 3 use only the model API ([`model_api.py`](model_api.py):
`load_model`, `context_states`, `predict_mixture`). They therefore run on any
model size and on `r1`, `layerwise`, `extra_node` and `reg_axis` alike. `r1`
is retired and still works, but the examples below use `layerwise` and
`extra_node`. Only diagnostic 1 and the two metadata-column probes of
diagnostic 2 are specific to `extra_node`.

## Data

Diagnostics 2 and 3 draw tasks from the training prior with the training
generator (`tasks.prior_task`), as `kshot-prior` does: task i has the key
`make_key(SOURCE_ADHOC, D, base_seed, i)`, which collides with no training,
validation or test stream. The default base seeds are 9201 (probes) and 9301
(edits), different from `kshot-prior` (9101) and the bank (4242).
`--data_config` takes a prior config, or a pre-simulated dataset config
(`configs/datasets/*.yaml`). For a dataset config its `prior_config` is
used, and the chunk files are never read. By default it is the checkpoint's
training prior (`model_api.training_prior_config_path`): the copy
`training_prior_config.<ext>` that `mixed_train` writes next to the
checkpoints, or, for runs trained from pre-simulated data, the prior of
their `presimulated_data.yaml`. Runs trained before that copy existed and
without pre-simulated data need `--data_config`.

The number of variables D comes from the checkpoint as well
(`model_api.checkpoint_num_variables`): the largest D of the training
prior, or the number of rows of the extra-node index embedding.
`load_model` builds the model for that D, as training did, and raises a
clear error if `--num_variables` disagrees (larger than the model's D, or
not one of the training prior's `num_variables`).
Results and data stay outside the repository; pass an output path elsewhere.

## 1. Embedding geometry (`extra_node` only, parameters only)

`metadata_embedding_geometry(model.encoder)` reads the trained e_INT
(`encoder.int_meta_embedding`) and e_j (`encoder.index_embedding`). It never
runs the model. Every encoder block is pre-norm, so attention sees
`LayerNorm(token)`, not the raw token. The function reports
quantities after the first block's input LayerNorm
(`encoder.encoder.sample_layers[0].norm1`, with its affine weights), and raw
tokens if a model were post-norm. The output is a JSON file:

| key | meaning |
| --- | --- |
| `int_meta_embedding_norm`, `index_embedding_norms` | ‖e_INT‖, ‖e_j‖ |
| `cosine_int_meta_vs_index` | cos(e_INT, e_j), raw |
| `single_target` / `two_targets` | one entry per target set S = {j} / {j, k} |
| `…raw_token_norms`, `…summed_index_embedding_norms` | ‖e_INT + e_S‖, ‖e_S‖ |
| `…projection_on_summed_index_embedding` | ⟨e_INT + e_S, e_S⟩ / ‖e_S‖ |
| `…normalised_cosine_to_observational` | cos(N(e_INT + e_S), N(e_INT)) |
| `…normalised_distance_to_observational` | ‖N(e_INT + e_S) − N(e_INT)‖ |
| `…min_pairwise_normalised_distance` | min over S ≠ S′ of ‖N(e_INT + e_S) − N(e_INT + e_S′)‖ |

```bash
python -m CITNP.evaluation embedding-geometry \
    --ckpt RUN/best_model.pt --model_config RUN/resolved_model_config.yaml \
    --out OUT/extra_node_metadata_embedding_geometry.json
```

**Reading it.** LayerNorm removes each token's mean and scale. If ‖e_INT‖ ≫
‖e_j‖, the normalised cosine to the observational token is close to 1, and
both normalised distances (to the observational token and between targets)
are small compared with the √d_model scale of normalised tokens. Intervened
and observational rows then enter the first attention layer almost
identically. A cosine well below 1 and clearly separated targets mean the
input encoding itself is not the problem. The `two_targets` block shows how
the summed embedding grows with |S|, although the current generator only
produces single-variable regimes. This diagnostic is about the input only;
later layers can amplify a small difference or erase a large one, which is
what diagnostic 2 measures.

## 2. Per-layer linear probes (any encoding)

`run_intervention_probes` takes `rows_per_task` context rows per task, half of
them intervened where possible. It reads their states at every depth of
`model_api.probe_points(model)`: `embed`, then `block<l>/sample` and
`block<l>/node` (`reg_axis`: `block<l>/sample` and `block<l>/feedback`). It
then fits a linear probe per depth:

| probe | states | label |
| --- | --- | --- |
| (a) `metadata_row_intervened` | metadata column (`extra_node`) | row intervened vs observational |
| (b) `metadata_intervention_target` | metadata column of single-target intervened rows (`extra_node`) | which variable (D classes) |
| (c) `cell_row_intervened` | every data cell | does my row contain an intervened cell? |
| (c′) `cell_row_intervened_excluding_targets` | data cells that are not themselves intervened | same; only row information that reached the other cells |
| (d) `cell_is_intervention_target` | every data cell | is this cell the intervened one? |

The metadata column comes from `context_states(..., include_metadata_node=True)`.
The probe is class-balanced softmax regression in PyTorch
(`probes.fit_classification_probe`), with features standardised on the
fitting folds. Folds are split by task: all rows of a task sit in the same
fold, never split by row. The CSV reports out-of-fold `accuracy`,
`balanced_accuracy` (mean per-class recall; classes are imbalanced, so use
this one), `auc` (binary probes) and `majority_class_rate`, all per depth.
Each probe comes twice: `labels = true` and `labels = shuffled`, which is the
same probe on one random permutation of its labels. The shuffled-label
control should sit at chance (0.5, or 1/D for (b)); the gap above it is the
signal. Cell probes also have a `value_baseline` row (depth −1) that uses only
the cell's standardised value and a one-hot of its variable. Results are
deterministic given `--seed`.

```bash
python -m CITNP.evaluation probe-intervention \
    --ckpt RUN/best_model.pt --model_config RUN/resolved_model_config.yaml \
    --data_config CITNP/experiments/configs/oracle_d3_iw_motif.yaml \
    --n_tasks 200 --out OUT/extra_node_intervention_probes.csv
```

**Reading it.** Plot `balanced_accuracy` against `depth` per probe, true
against shuffled.

- (a) decays with depth while (b) stays high: the metadata column still
  carries the target identity, but the "intervened vs observational"
  direction gets absorbed into the observational representation. This is the
  failure mode the question is about.
- (a) and (b) both stay high: the metadata column keeps the information.
  (c′) then shows whether it reaches the data cells, which is the only route
  to the prediction (the column is dropped before the head).
- (c′) at the shuffled level at every depth: row-level intervention
  information never spreads from the metadata column (or flag) to the other
  cells of the row.
- (d) far above `value_baseline`: the representation encodes "this cell was
  set", beyond what its value alone gives away. Do-values often lie outside
  the observational range, so the baseline is not at chance.
- For `layerwise` and `reg_axis` the flag is re-injected at every block, so
  (c) and (d) should not decay. That gives a reference curve for
  `extra_node`.

A linear probe gives a lower bound on what is encoded, and high accuracy
does not mean the model uses the information. Diagnostic 3 tests use.

## 3. Label sensitivity under counterfactual mask edits (any encoding)

The question is whether the model responds to the intervention label, not
whether it recovers the truth. `run_context_edits` edits only the
intervention mask of a context; the values never change. The edited context
is therefore deliberately mislabelled. The reference is the exact Bayesian
oracle given the same edited context (`LinGaussOracle` refitted on (x,
edited mask)), i.e. a reasoner who believes the labels. Each predictor is
compared with its own prediction on the unedited context. Per task, a seeded
order of the intervened rows is drawn; with n edited rows, the first n rows
of that order are edited, for n on the geometric grid 1, 2, 4, … and always
all intervened rows (`edit_counts`):

- `relabel_observational`: the n rows are relabelled observational;
- `move_target`: each of the n rows has its intervention moved from variable
  j to a seeded variable k ≠ j. Any k is allowed, the query's treatment and
  outcome included, because neither model nor oracle needs a restriction.

The edited mask goes through the model's normal input path
(`context_int_mask`). The model also standardises each context with its
observational rows. Relabelling rows observational therefore changes two
things at once: the label, and the scaling of every value (and, for
`scale_conditioning: stats` models, the scale features `context_scale`).
Run both settings:

- **`--keep_original_normalisation` (primary readout).** Every edited
  version gets the unedited context's standardisation, its scale features
  and its map back to the raw scale (`predict_mixture`,
  `normalisation_int_mask`). Only the intervention mask differs from the
  unedited input, so any change in the prediction is a response to the
  label. Use this run to answer the encoding question.
- **Default (normal input path).** The standardisation comes from the
  edited mask, as it would for a context that really was labelled that way.
  Compare it with the primary run to see how much of the response comes
  from the rescaling alone.

`move_target` never changes the standardisation (observational rows stay
observational), so the two runs agree for it up to Monte Carlo noise.

Queries are each task's own query rows: their do-values t (at most
`--max_query_points` per treatment) and every outcome Y ≠ T. Per (task, T, Y,
edit, n), for model and oracle:

| column | meaning |
| --- | --- |
| `*_mean_shift`, `*_mean_shift_rms` | edited − original predictive mean: mean and RMS over t |
| `*_slope_change` | change of the predictive mean's slope in t |
| `*_kl_original_to_edited` | KL(original ‖ edited) predictive, Monte Carlo (`n_y` draws per t from the original; the same draws for every n), mean over t |
| `*_kl_*_se` | Monte Carlo standard error of each KL over the y draws (`metrics.kl_mc_mean_with_standard_error`); it does not include the oracle's own Monte Carlo error over parameter draws (`--n_mc`) |
| `mean_shift_sign_agreement` | fraction of t where the model's shift has the oracle's sign |
| `secondary_kl_oracle_to_model_original` / `_edited` | KL(oracle ‖ model) on the unedited / edited context; **secondary only** |
| strata | `identifiable_from_observational` (population check on the true SCM: every DAG in its Markov equivalence class gives the same p(X_Y \| do(X_T))), `treatment_intervened_in_context`, `outcome_intervened_in_context`, `relation`, `num_edited_rows_targeting_treatment`; for `move_target` also `new_target_role` (`treatment`, `outcome` or `other`: the new target k relative to the query; `mixed` when the n moved rows differ; `not_applicable` for relabelling) with `num_moved_to_treatment` / `_outcome` / `_other` |

The predictives are mixtures, so there is no closed-form KL. The KLs are
Monte Carlo estimates, the way `kshot` / `kshot-prior` compare mixtures
(`metrics.kl_mc`). The ground truth's log-likelihood is not used.

`<out>_summary.csv` (`summarise_context_edits`) groups the rows by edit, by
each combination of the strata `identifiable_from_observational`,
`treatment_intervened_in_context` and `new_target_role`, and by
`num_edited_rows` or not.
`all` marks an ungrouped key. Per group it reports `mean_shift_ratio`,
`slope_change_ratio` and `kl_ratio` (model / oracle of the mean magnitudes);
`mean_shift_correlation`, `slope_change_correlation` and `kl_correlation`
(Pearson correlation over queries); `mean_shift_sign_agreement` and
`slope_change_sign_agreement`; and `*_se`, the Monte Carlo standard error of
each mean KL. That SE treats rows as independent: true across queries, but
not across the n of one query (they reuse the same y draws), so it is too
small for groups that pool several n. The rows with `num_edited_rows ≠ all` and
every stratum `all` form the shift-vs-number-of-edited-rows curves. Their x axis is
the absolute number of edited rows. The last grid point of each task is all
of its rows, a number that differs between tasks, so for the "everything
relabelled" end point select rows with `fraction_edited_rows == 1` in the
per-query CSV.

```bash
# primary readout: only the intervention mask changes
python -m CITNP.evaluation context-edits \
    --ckpt RUN/best_model.pt --model_config RUN/resolved_model_config.yaml \
    --data_config CITNP/experiments/configs/oracle_d3_iw_motif.yaml \
    --n_tasks 200 --keep_original_normalisation \
    --out OUT/extra_node_context_edits_original_normalisation.csv
# normal input path (standardisation from the edited mask)
python -m CITNP.evaluation context-edits \
    --ckpt RUN/best_model.pt --model_config RUN/resolved_model_config.yaml \
    --data_config CITNP/experiments/configs/oracle_d3_iw_motif.yaml \
    --n_tasks 200 --out OUT/extra_node_context_edits_edited_normalisation.csv
```

**Reading it.** Read the `original_normalisation` run first; the points
below refer to it. The `edited_normalisation` run differs only for
`relabel_observational`.

- A ratio near 1 with high correlation and sign agreement: the model reacts
  to the labels as much as, and in the same direction as, a reasoner who
  believes them.
- A ratio near 0: the model ignores the labels. If this holds for
  `relabel_observational` but not for `layerwise`, the intervened rows are
  being treated as observational.
- `identifiable_from_observational` is a property of the true graph and
  weights (`tasks.observationally_identified`): on a large observation-only
  context the oracle is then certain of the effect, and otherwise it stays a
  mixture over the Markov-equivalent DAGs. It ignores the context's own
  interventional rows, which is what `treatment_intervened_in_context`
  stratifies. Graph priors with zero mass on some equivalent DAGs (motif
  priors) can make the oracle certain of a query labelled unidentifiable.
- Ratios are most informative where the oracle moves. The oracle moves most
  for queries that are not identifiable from observational data alone and
  when rows intervening on the query treatment are edited
  (`num_edited_rows_targeting_treatment`). Identifiable queries need no
  interventions, so small shifts there are expected for both.
- For `move_target`, compare `new_target_role` at `num_edited_rows = 1`:
  the ratio and correlation per role show whether the model's response
  depends on which variable the label names in the same way as the
  oracle's does.
- The curves show whether the model's response grows with the number of
  relabelled rows the way the oracle's does. A flat model curve under a
  rising oracle curve means insensitivity to the labels.
- `secondary_kl_oracle_to_model_edited` measures the model-vs-oracle gap on
  contexts that are **outside the training distribution** (mislabelled
  values, e.g. do-values presented as observational). A large value there is
  expected and does not say how the model does on real contexts. Use
  `kshot-prior` for that.

Requirements: a linear-Gaussian prior with `normalise: false`
(`tasks.require_exact_oracle`) and D ≤ 4. Cost: the grid has about
2 (log2 R + 1) edited versions per task for R intervened rows. All versions
of a context and all its queries go through the model in one forward pass
(`model_api.predict_mixture_versions`; `--max_batch_size` splits it if
memory runs out), and the oracle is refitted once per version. Split long
runs over tasks with `--task_offset`.

## Running on BMRC

[`eval_intervention_diagnostics.sh`](../experiments/bmrc/eval_intervention_diagnostics.sh)
runs all three diagnostics for one checkpoint, with `context-edits` in both
normalisation settings (`<OUT_PREFIX>_context_edits_original_normalisation*`,
the primary readout, and `<OUT_PREFIX>_context_edits_edited_normalisation*`).
It skips embedding-geometry when `MODEL_CFG` is not `extra_node`.
`DRY_RUN=1` prints the commands.

```bash
mkdir -p /gpfs3/well/cebam/users/$USER/alcfm/logs
for MODEL in extra_node layerwise; do
  RUN=<results root>/<run name of $MODEL>
  REPO=$PWD CKPT=$RUN/best_model.pt MODEL_CFG=$RUN/resolved_model_config.yaml \
      OUT_PREFIX=<diagnostics dir>/$MODEL \
      sbatch CITNP/experiments/bmrc/eval_intervention_diagnostics.sh
done
# runs that record no training prior: add DATA_CFG=<the prior they were trained on>
```

Tests: `testing/test_context_states.py`, `test_embedding_geometry.py`,
`test_intervention_probes.py`, `test_context_edits.py` (tiny random models,
CPU; they check structure, not learned behaviour).
