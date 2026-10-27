# Mixed-context MACE-TNP training (design R1)

The model predicts **p(X_Y | do(X_T = t), C)**, where the context C mixes
observational rows and rows from interventional experiments.

## Design R1: one upstream question per group, plus an intervened flag

Each training task (grouped query mode, see
[`MIXED_REGIME_README.md`](../datasets/MIXED_REGIME_README.md#grouped-query-mode))
carries G question groups. Group g has its own treatment T_g, outcome Y_g and
block of query rows. Every group is exactly **one upstream MACE-TNP question**:

1. `prepare_mixed_batch` (`CITNP/trainer/mixed_context_trainer.py`) flattens
   (task, group) into the batch dimension, duplicating the task's context for
   each group. T_g is read from the query rows; Y_g is drawn uniformly from
   the other variables (a fresh draw per training step; a fixed hash of the
   task key for validation and test). Each task is standardised with the mean
   and std of its non-intervened context cells.
2. `MixedContextMoGModel` (`CITNP/models/mixed_context_model.py`) calls
   upstream's role-based embedding unchanged, with `intervention_index = T_g`
   and `outcome_index = Y_g` per flattened element.
3. The only new component: an **intervened-flag embedding**
   (`encoder.context_intervened`), added to context cells that were set by an
   experiment (`ctx_int`). Its last layer starts at zero.

With G = 1 and no intervened cells, the model is exactly upstream MACE-TNP.

## Smoke test (D = 3, a few minutes on CPU)

```bash
python -m CITNP.experiments.smoke_mixed_d3 [--device cuda] [--num_workers 2]
```

Trains 3 epochs × 50 iterations (batch 16) on
[`configs/smoke_d3.yaml`](configs/smoke_d3.yaml) with the model in
[`configs/smoke_d3_model.yaml`](configs/smoke_d3_model.yaml) (d_model 64, 4
layers, MHCA, 3 mixture components), then checks: no NaN/inf losses; training
loss falls; validation loss falls over the epochs. It also prints a
uses-interventions check on 200 validation tasks: the NLL with only the
observational context rows minus the NLL with the full context, per group (in
the data's original scale, since the two contexts are standardised
differently). It should be ≥ 0 on seen groups. Exit status 1 if a check fails.

## Full training

```bash
python -m CITNP.experiments.mixed_train \
    --data_config <grouped MixedRegimeConfig yaml> \
    [--model_config <yaml>] [--init_from_upstream <ckpt>] \
    --batch_size 32 --iterations_per_epoch 1000 --epochs 100 \
    --num_workers 8 --seed 0 --device cuda --dtype bfloat16
```

All upstream arguments (`CITNP/utils/default_args.py`) are accepted.
`--model_config` overrides model arguments (`d_model`, `nhead`,
`num_layers_encoder`, `sample_attn_mode`, `dim_feedforward`, `emb_depth`,
`num_mixture_components`, ...). Data:

- training: `MixedRegimeTaskDataset(split="train")`, fresh tasks every epoch
  (`SetEpochCallback` is added for non-persistent workers);
- validation / test: `split="val"` / `"test"` with `fresh_each_epoch=False`,
  `--val_iterations` / `--test_iterations` batches, identical on every pass.

The data config must use `qry_mode: grouped`. Upstream's wandb validation
plots are disabled (they expect upstream's tuple batches). The test epoch
reports losses overall and by stratum: causal direction of T → Y, seen /
unseen group, `has_obs`, K (0, 1, 2+) and G.

## Warm start from upstream

`--init_from_upstream <ckpt>` loads an upstream MACE-TNP checkpoint
(`CheckpointCallback` format, or a raw state dict) with
`load_upstream_checkpoint`. Every parameter is loaded except
`encoder.context_intervened.*`, which keeps its zero-initialised last layer,
so the warm-started model makes exactly the upstream predictions until the
flag embedding is trained. Any other missing or unexpected key raises.

## Alternative encoding: `extra_node`

`--intervention_encoding extra_node` (or `intervention_encoding: extra_node` in
the model YAML, e.g.
[`configs/oracle_d3_model_m_extra_node.yaml`](configs/oracle_d3_model_m_extra_node.yaml))
replaces R1's per-cell flag by one extra node position per row. For D
variables the transformer runs on D + 1 node positions
`[X_0, ..., X_{D-1}, META]`:

```
context cell     h_ij      = f_C,r_j(x_ij) + e_j
context metadata h_i,META  = e_INT + sum_j m_ij e_j
query cell       h_qj      = f_Q,r_j(x~_qj) + e_j
query metadata   h_q,META  = f_Q,variable(0)        (no index embedding)
```

- `f_C,r_j` / `f_Q,r_j` are upstream's six role MLPs (role r_j from the
  question's T and Y); `x~` is the query row with everything but the do-value
  masked, as upstream.
- `e_j` (`encoder.index_embedding`) are learned variable-index embeddings,
  shared by variable and metadata tokens; `e_INT`
  (`encoder.int_meta_embedding`) is a learned metadata-type embedding. Both
  start at normal(0, 0.02).
- `m_i` is the row's `context_int_mask`; `None` means observational, i.e. the
  metadata token is `e_INT`.
- The metadata position is dropped before the prediction head, so everything
  downstream is unchanged. There are no R1 or layer-wise parameters.

Flag ablation (`zero_flags=True` in `predict_mixture` / `context_states`)
sets every `m_i` to zero, so every metadata token is `e_INT`.

The extra-node parameters are not zero-initialised, so `--init_from_upstream`
loads but does not reproduce upstream at step 0, and `--init_from_r1` is
rejected. For the comparison with layerwise, train from scratch.

## Scale conditioning (information parity with the oracle)

Per-task standardisation removes each variable's scale from the context, but
the exact oracle sees the raw context. `--scale_conditioning stats` (or
`scale_conditioning: stats` in the model YAML; default `none`) gives the model
the removed statistics: per variable j the pair (log std_j, mean_j / std_j),
batch key `context_scale` of shape [B·G, D, 2] (zeros when `normalise: false`).
Standardised context plus these statistics recovers the raw context:
`x_raw = x_std * exp(log std) + (mean / std) * std`.

An MLP `ScaleConditioning` (2 → `d_model`, depth `emb_depth`, last Linear layer
zero-initialised) maps the pair to an embedding added to every row (context and
target) of variable column j, right after the row embeddings are assembled and
before the transformer stack. It works with all three encodings; with
`extra_node` the INT_META column gets nothing. With `none` no parameters are
created, so existing checkpoints load strictly. Configs: `*_stats.yaml`.

## Training from pre-simulated datasets

Instead of generating tasks on the fly, a run can read tasks that were
simulated once and stored as chunk files. **The model's inputs are identical to
streaming**: both paths draw the batch shape with `draw_batch_shape` and
generate each task with `generate_task_with_shape`, and the normalisation, the
variance information of `scale_conditioning: stats` (`context_scale`), the
outcome draws and the intervention encodings are computed from the loaded batch
by the same `prepare_mixed_batch` code (`testing/test_presimulated_data.py`
checks this tensor by tensor). Only the source of the tasks differs. Streaming
stays the default.

A dataset config in `CITNP/experiments/configs/datasets/` names the prior and
the split sizes (see `CITNP/datasets/dataset_config.py`). Simulated data live
outside the checkout, under `$DATA_ROOT` (default `$OUT/data`, exported to
Python as `ALCFM_DATA_ROOT`):

```bash
export DATASET_CONFIG=CITNP/experiments/configs/datasets/oracle_d3_iw_motif__v1.yaml
SPLIT=train      sbatch CITNP/experiments/bmrc/generate_presimulated_data.sh
SPLIT=validation sbatch CITNP/experiments/bmrc/generate_presimulated_data.sh
SPLIT=test       sbatch CITNP/experiments/bmrc/generate_presimulated_data.sh
DATASET_CONFIG=$DATASET_CONFIG MODEL_CFG=CITNP/experiments/configs/oracle_d3_model_m_extra_node_stats.yaml \
    sbatch CITNP/experiments/bmrc/train.sh
```

`generate_presimulated_data.sh` is a fixed-size CPU array (job `i` of `NUM_JOBS`
writes chunk files `i, i + NUM_JOBS, ...`) and skips files that already exist,
so resubmitting after a failure only fills the gaps. The writer refuses to add
files to a directory whose `config.json` has other generation settings.

`mixed_train.py --dataset_config <yaml> [--data_root DIR]` trains from the
chunks (`--data_config` must be the dataset's prior; the run aborts if the
chunks were written from another one). Training uses
`MixedRegimeChunkDataset(iterations_per_epoch=...)`, so an epoch has the same
number of batches as when streaming; each pass over the data reshuffles files
and tasks. A validation or test split that is not in the dataset config or not
generated falls back to streaming; for chunk splits `--val_iterations` and
`--test_iterations` are ignored (one pass over the split). The run directory
records the dataset config, the split directories and the fingerprints of their
`config.json` files in `presimulated_data.yaml`, and `train.sh` refuses to
resume if the training data changed. With `--resume_full` the data order is
re-seeded from the resumed epoch rather than restored exactly.

**Sizing.** The start prints the unique training tasks, the tasks the run
consumes (`epochs x iterations_per_epoch x batch_size`) and the implied number of
passes. `oracle_d3_iw_motif__v1` has 2,048,000 training tasks; the default
500 epochs x 200 iterations x 256 tasks consume 25,600,000, i.e. 12.5 passes, so
a task is seen about 12 times (streaming never repeats one). Check the bytes per
task with `du -sh` on a small split before sizing a large one.

## Resuming a run

`--resume_full <results/RUN/last_model.pt>` continues that run in its own
results directory (any other existing run directory is still refused): strict
model weights, optimiser and scheduler state, global step, epoch and best
validation loss are restored, and the data stream continues from the next
epoch. Use the same configs and `--epochs`. Only epoch-boundary checkpoints
(`last_model.pt`) can be resumed; a mid-epoch `step_N.pt` is rejected.
`bmrc/train.sh` adds this flag automatically when `last_model.pt` exists and the
recorded config hashes match. Checkpoints written before this change lack the
best validation loss, so the first validation after resuming may overwrite
`best_model.pt`.

`--fine_tune_from <ckpt>` (formerly `--resume_from`) is the other half: it starts
a NEW run from a checkpoint's model weights only (strict match, fresh optimiser,
step 0, new warmup and cosine schedule, new run directory). The two flags are
mutually exclusive.

## Information-parity grid (2 x 2)

Question: does a model that sees only the standardised context lose anything
the exact oracle uses, and does `scale_conditioning: stats` recover it?
`bmrc/submit_parity_grid.sh` submits four runs with the same seed, steps and
model size (extra-node encoding by default):

| Run | Data config | Model config |
|---|---|---|
| A | `oracle_d3_iw_motif.yaml` | `oracle_d3_model_m_extra_node.yaml` |
| B | `oracle_d3_iw_motif.yaml` | `oracle_d3_model_m_extra_node_stats.yaml` |
| C | `oracle_d3_nig_eta9_motif.yaml` | `oracle_d3_model_m_extra_node.yaml` |
| D | `oracle_d3_nig_eta9_motif.yaml` | `oracle_d3_model_m_extra_node_stats.yaml` |

The IW prior (`linear_iw`) is score-equivalent, so scale carries no information
about edge orientation; `oracle_d3_nig_eta9_motif.yaml` (`linear_nig`,
`weight_scale: 9`) is not, so orientation is partly identifiable through scale.
Each data config has its own oracle bank
(`$OUT/banks/<config>__<sha1>.pkl`, built with `LinearGaussianPrior.from_config`
and the exact `motif_graph_prior`), and each run is scored on its own prior's
bank.

**Expected results**
- **A ~ B:** under the IW prior scale carries no orientation information, so
  stats conditioning should not help. In the `nT=0` bucket, the gap to the
  oracle is larger for C than for D (the oracle uses scale, C's model cannot).
- **If B clearly beats A,** something leaks scale or graph information that the
  oracle does not use: recheck the score-equivalence test
  (`test_score_equivalence.py`) and the graph-independence guard
  (`test_motif_prior.py`).

**Metrics**
- Excess NLL over the oracle: `oracle_gap/*` in each run's `oracle_gap.csv`
  (by `nT` bucket, `K` and `has_obs`), logged every `--oracle_every` steps.
- Interval calibration: `calib/cover90` in the same file.
- Per question class, PIT histograms and the `nT=0` divergence from the
  oracle's predictive need the true graph and the full predictive. The bank does
  not store them; compute these with `predict_mixture` / `mixture_cdf`
  (`evaluation/model_api.py`) and the oracle's `predictive_components`, as
  `evaluation/oracle_calibration.py` does for the oracle.
