# `reg_axis_full`: regime-aware encoding without compression

`reg_axis_full` is the fifth intervention encoding of the mixed-context
MACE-TNP model, next to `r1`, `layerwise`, `extra_node` and `reg_axis`
([`mixed_context_model.py`](../mixed_context_model.py)). Like
[`reg_axis`](../regime_axis/README.md) it treats each experimental regime as
its own unit. Unlike `reg_axis` it does not compress a regime into memory
tokens: every reader attends to every regime's full set of rows, one regime at
a time, and then combines the per-regime results with a second attention over
regimes. Everything else is reused unchanged: the six role MLPs, the target
masking, the prediction head, the losses, the trainer, the data, scale
conditioning, and from `reg_axis` the regime grouping and the flag.

Notation: B = tasks × query groups (the flattened batch), Nc context rows,
Nq query rows, L reader rows (Nc or Nq), D variables, d = `d_model`,
H = `nhead`, R = R_max regimes in the batch, n_s rows of regime s,
N = max n_s in the batch.

## Motivation

The context is a union of experiments: an observational sample and samples
from do(X_S = ·) for several target sets S. `reg_axis` summarises each
experiment in K memory tokens per variable before experiments exchange
information, so what a query can learn about an experiment is limited to what
fits in K tokens, chosen before the query is known. `reg_axis_full` removes
that bottleneck. A reader (a context cell or a query cell) forms its own
summary of each experiment, by attending over that experiment's rows with its
own query vector, and then weighs the experiments against each other. The
regime structure is still built in: the softmax of the first stage never mixes
rows of different regimes.

## Regimes

A regime is the set of context rows that share one intervention-target set
(row i of `context_int_mask`); intervention values never matter.
`group_regimes` from [`regime_axis/grouping.py`](../regime_axis/grouping.py)
computes the regimes inside the encoder, so the model input is unchanged:
flat `context_data` [B, Nc, D, 1] and `context_int_mask` [B, Nc, D], rows in
any order. Slots are bookkeeping and are never embedded, so the model is
invariant to row order and to slot order. The optional `context_regime`
labels are a consistency check only.

## Two-stage regime attention ([`attention.py`](attention.py))

For one variable j and one reader token x. Keys are always context cells of
the same variable j; h[s, k] is the cell of variable j in row k of regime s.

**Stage 1**, for each source regime s (softmax over the rows k of regime s
only):

```text
alpha[s,k] = softmax_k( <W_Q x, W_K h[s,k]> / sqrt(d/H) )
m[s]       = sum_k alpha[s,k] * W_V h[s,k]
             + count(log1p(n_s))                         (reg_axis_count_feature)
             + e_own * 1[s is the reader's own regime]   (context readers only)
```

**Stage 2** (softmax over the regimes present in the task):

```text
w[s] = softmax_s( <U_Q x, U_K norm(m[s])> / sqrt(d/H) )
x   <- x + W_O sum_s w[s] * m[s]
x   <- x + FFN(x)
```

`TwoStageRegimeAttentionLayer` follows the conventions of
`MaskedSetAttentionLayer`: pre-norm (one LayerNorm shared by readers and
keys), residual, GELU FFN of width `dim_feedforward`, dropout, bias-free
q/k/v projections, output projection `W_O` with dropout. Stage 1 has H heads,
concatenated to a message m[s] of width d. Stage 2 has H heads as well; its
value is m[s] itself (no value projection), and `norm` is a LayerNorm applied
to the messages for scoring only. `count` is `Linear(1, d)`. `e_own` is a
learned vector of size d, initialised at zero, and exists only in layers whose
readers are context cells.

| parameter | shape | role |
| --- | --- | --- |
| `to_q`, `to_k`, `to_v` | d × d, no bias | W_Q, W_K, W_V (stage 1) |
| `count` | 1 → d | size of the source regime (optional) |
| `e_own` | d | own-regime marker (context read only) |
| `regime_q`, `regime_k` | d × d, no bias | U_Q, U_K (stage 2) |
| `regime_norm` | LayerNorm(d) | `norm` on the messages, scoring only |
| `to_out` | d → d | W_O |
| `norm1`, `norm2`, `ff_block` | — | pre-norms and FFN |

## Input encoding ([`encoder.py`](encoder.py))

Identical to `reg_axis`
([`regime_axis/README.md`](../regime_axis/README.md#input-encoding-encoderpy-encodingpy)):

```text
h0 = role MLP embedding                                  (upstream embed())
   + flag * flag_embed(x)    on intervened context cells (target_flag injected)
   + ScaleConditioning(stats_j) on every row of column j (scale_conditioning stats)
```

`RegimeAxisFullEncoder` subclasses upstream `CausalInfEncoder`, keeps the six
role MLPs and `embed()`, deletes the inherited transformer stack
(`self.encoder`) and builds `self.blocks`. The flag is `FlagInjection` from
`reg_axis`, unchanged, with parameters under `encoder.flag.`; with
`reg_axis_target_flag: none` there are no flag parameters and regimes are
known only through the grouping. Query rows carry only the treatment value
(upstream `_mask_target_data`) and never get a flag.

## One block ([`block.py`](block.py))

Context tokens `hc` [B, Nc, D, d]; query tokens `hq` [B, Nq, D, d].

| step | operation | readers | keys | mask (True = attend) |
| --- | --- | --- | --- | --- |
| 0 | `hc += inject_var[l](code)` | — | — | — |
| 1 | variable attention over the D cells of each row; context and query rows, shared weights | [B·(Nc+Nq), D, d] | same | none |
| 0' | `hc += inject_sample[l](code)` | — | — | — |
| 2 | context read, with `e_own` | `hc` [B, Nc, D, d] | `hc` after 0' | stage 1: `key_mask` [B, R, 1, 1, N]; stage 2: `slot_present` [B, 1, 1, 1, 1, R] |
| 3 | query read, without `e_own` | `hq` [B, Nq, D, d] | **updated** `hc` from step 2 | as step 2 |

Tensor shapes inside steps 2 and 3:

| tensor | shape |
| --- | --- |
| stage-1 queries | [B, 1, D·H, L, d/H] (broadcast over R) |
| stage-1 keys and values | [B, R, D·H, N, d/H] |
| messages m | [B, D, L, R, d] |
| stage-2 queries | [B, D, L, H, 1, d/H] |
| stage-2 keys (`regime_k(norm(m))`) and values (m) | [B, D, L, H, R, d/H] |

```text
          context hc [B,Nc,D,d]                     query hq [B,Nq,D,d]
                 │ (+inject_var)                           │
                 └──────────► 1. variable attention ◄──────┘   (shared, per row)
                 │ (+inject_sample)                        │
   2. context read: each context cell                      │
      attends to every regime's rows, then                 │
      over the regimes (+count, +e_own)                    │
                 │                                         │
                 ├───── updated hc (keys) ─────► 3. query read: each query cell
                 │                                  attends to every regime's
                 ▼                                  rows, then over the regimes
            next block                                     ▼
                                                   next block / head
```

Steps 2 and 3 are two layer instances with separate weights. Every block has
both, including the last: the context update of the last block reaches the
prediction through the query read of the same block, so no parameter is
unused (DistributedDataParallel needs no `find_unused_parameters`).

After the last block the query tokens go through upstream
`_extract_outcome_representation(rep, outcome_index, num_trgt=Nq)` and the
upstream predictor, exactly as in `reg_axis`.

## Packing ([`packing.py`](packing.py))

Readers stay flat; only keys are packed. `pack_regimes(grouping)` builds a
`RegimePacking` once per forward pass, without Python loops over tasks or
rows (a stable sort of the rows by slot, then one scatter):

| field | shape | meaning |
| --- | --- | --- |
| `index` | [B, R, N] long | row index of the k-th row of regime r; 0 at padded positions |
| `valid` | [B, R, N] bool | real rows; `valid.sum(-1) == n_r` |
| `key_mask` | [B, R, N] bool | stage-1 mask: `valid \| ~slot_present[..., None]` |
| `slot_present` | [B, R] bool | stage-2 mask |
| `own` | [B, Nc, R] bool | regime r is the regime of context row i |
| `n_r` | [B, R] long | rows per regime |

`gather_keys(tokens, index)` maps context tokens [B, Nc, D, d] to packed keys
[B, R, N, D, d]. The projected keys and values are gathered, moved to
[B, R, D·H, N, d/H], and stage 1 is **one**
`F.scaled_dot_product_attention` call with the source regime as an extra
leading dimension: the queries broadcast over R and the boolean key mask
broadcasts over variables, heads and readers. The attention-weight tensor is
never built by hand and there is no loop over regimes.

## Absent regimes

With `--same_numregimes_per_batch` every task of a training batch has R
regimes and nothing is absent. The model still accepts batches with mixed
regime counts (evaluation, or training without the flag):

- Stage 1 never sees an all-False mask row. The keys of an absent regime are
  left readable (`key_mask = valid | ~slot_present[..., None]`; its padded
  positions all point at row 0), so its message is finite but meaningless.
- Stage 2 excludes absent regimes with `slot_present`, so that message gets
  weight exactly zero and no gradient. Every task has at least one regime, so
  stage 2 always has a valid key.

The model does not rely on PyTorch returning zeros for fully masked attention
rows. A task's predictions are the same alone and inside a batch whose other
tasks have other regime counts and sizes (`test_batch_independence`).

## Query conditioning and no leakage

- Query rows carry only t. They are mixed with other variables of the same
  row (step 1) and read context rows (step 3).
- Query rows are never keys, never attend to each other, and are never read
  by context rows. Predictions therefore do not depend on the hidden query
  values, and adding a query row leaves the other predictions unchanged
  (`test_query_leakage`).
- Context is re-encoded once per query group, because the role embedding
  depends on (T, Y).

## Set size

Both stages are softmax averages, so without the count feature duplicating
every row of a regime changes nothing (`test_size_neutrality`). With
`reg_axis_count_feature: true` (default) `Linear(1, d)(log1p(n_s))` is added
to the message of regime s, in the context read and in the query read (one
Linear per layer).

## Configuration

| key (model YAML / `CausalInfModelConfig`) | default | meaning |
| --- | --- | --- |
| `intervention_encoding: reg_axis_full` | — | selects `RegimeAxisFullMixedContextMoGModel` |
| `reg_axis_num_blocks` | `null` → `num_layers_encoder // 2` | number of blocks |
| `reg_axis_target_flag` | `injected` | `injected` or `none` |
| `reg_axis_count_feature` | `true` | count feature on the messages |

The three keys are shared with `reg_axis` and stay rejected for `r1`,
`layerwise` and `extra_node`. `reg_axis_num_memory_tokens` is an error with
`reg_axis_full` (there are no memory tokens). `sample_attn_mode` is ignored.
`linear_attention: true` is rejected, because the regime masks need softmax
attention. `variable_mask` is not supported. `scale_conditioning` works
exactly as for the other encodings. `--init_from_upstream` is rejected: the
transformer parameters have no upstream counterpart. Configs:
[`smoke_d3_model_reg_axis_full.yaml`](../../experiments/configs/smoke_d3_model_reg_axis_full.yaml)
and
[`oracle_d3_model_m_reg_axis_full_stats.yaml`](../../experiments/configs/oracle_d3_model_m_reg_axis_full_stats.yaml),
each identical to its `reg_axis` counterpart apart from the encoding and the
removed memory-token key.

**Train with `--same_numregimes_per_batch`.** The packed keys have R·N
positions per task, where R and N are batch maxima. Without the flag almost
every batch contains a task with the maximum number of regimes, so every task
pays for it; with the flag all tasks of a batch have the same R (see
[Cost](#cost)). The flag changes only which tasks share a batch, not the
prior over tasks
([`MIXED_REGIME_README.md`](../../datasets/MIXED_REGIME_README.md#same-number-of-regimes-per-batch)).

Smoke run (CPU):

```bash
python -m CITNP.experiments.mixed_train \
    --data_config CITNP/experiments/configs/smoke_d3.yaml \
    --model_config CITNP/experiments/configs/smoke_d3_model_reg_axis_full.yaml \
    --same_numregimes_per_batch \
    --batch_size 16 --iterations_per_epoch 50 --epochs 3 \
    --num_workers 0 --seed 0 --device cpu --no_wandb \
    --run_name smoke_reg_axis_full
```

Full training: the same command as for the other encodings
([`MIXED_TRAINING_README.md`](../../experiments/MIXED_TRAINING_README.md)),
with `--model_config CITNP/experiments/configs/oracle_d3_model_m_reg_axis_full_stats.yaml`
and `--same_numregimes_per_batch` (on BMRC: `SAME_NUMREGIMES_PER_BATCH=1`).
`model_api.load_model` rebuilds the model from the run's
`resolved_model_config.yaml`.

**`zero_flags` caveat.** With `zero_flags=True` in `predict_mixture`,
`context_int_mask` is all False. As for `reg_axis`, this also removes the
regime grouping: every row then falls into one observational regime, and the
two-stage attention reduces to plain attention over all context rows. The
ablation therefore removes both the flag and the regime structure, which is a
stronger change than for `r1` / `layerwise` (flag only) or `extra_node`
(metadata only).

`model_api.context_states` and the flag probe built on it hook the upstream
transformer stack (`model.encoder.encoder`), which neither regime-axis
encoding has; they do not support `reg_axis` or `reg_axis_full`.

## Cost

Per block and per read, with L readers:

| | time and attention-weight memory |
| --- | --- |
| stage 1 | B · D · H · L · R · N |
| stage 2 | B · D · H · L · R |
| messages | B · D · L · R · d floats |

The context read has L = Nc, the query read L = Nq. Full attention over the
context would cost B · D · H · L · Nc, so the overhead of packing is the
ratio R·N / Nc ≥ 1 of packed key positions to real rows. Padding comes from
two sources: tasks with fewer regimes than the batch maximum (removed by
`--same_numregimes_per_batch`) and regimes smaller than the largest regime of
the batch (not removed: the Dirichlet split makes regime sizes very unequal,
so in a large batch N is close to Nc).

Measured on the oracle D = 3 prior (`oracle_d3.yaml`, 50 streaming batches,
seed 0):

| batching | batch size | R·N / Nc: mean | max |
| --- | --- | --- | --- |
| `--same_numregimes_per_batch` | 16 | 1.69 | 3.60 |
| `--same_numregimes_per_batch` | 256 | 1.82 | 3.87 |
| default (R per task) | 256 | 4.00 | 4.00 |

With the flag the ratio is about R (1.00 for R = 1, about 2 for R = 2, …),
and R = 1 is the most likely value under this prior (P = 0.53).

Forward + backward of one training batch on CPU (14 threads, float32,
`oracle_d3_model_m*_stats.yaml` models, batch size 16, the same batch for
every model; median of 5 steps and peak resident memory of the process):

| batch (B·G, Nc, R, N) | layerwise | reg_axis | reg_axis_full |
| --- | --- | --- | --- |
| 32, 169, R = 1, N = 169 | 0.71 s, 1.7 GB | 0.92 s, 2.1 GB | 1.02 s, 2.8 GB |
| 48, 95, R = 2, N = 94 | 0.61 s, 1.6 GB | 0.73 s, 1.9 GB | 1.10 s, 2.4 GB |
| 48, 141, R = 3, N = 135 | 0.88 s, 1.9 GB | 1.31 s, 2.4 GB | 2.39 s, 4.5 GB |
| 32, 196, R = 4, N = 142 | 1.01 s, 1.9 GB | 1.17 s, 2.4 GB | 2.90 s, 4.5 GB |

So `reg_axis_full` costs about 1.4 times `layerwise` for single-regime
batches and 2.5 to 3 times for R = 3 or 4, in time and in memory.

Parameter counts (MoG models, D = 3; both regime-axis encodings with the
default `reg_axis_num_blocks` = `num_layers_encoder // 2`):

| config | r1 | layerwise | reg_axis | reg_axis_full |
| --- | --- | --- | --- | --- |
| `smoke_d3_model*.yaml` (d 64, 4 layers) | 238,601 | 255,177 | 426,441 | 293,193 |
| `oracle_d3_model_m*.yaml` (d 128, 6 layers) | 1,342,360 | 1,441,048 | 2,516,504 | 1,657,496 |
| `oracle_d3_model_m*_stats.yaml` | 1,359,256 | 1,457,944 | 2,533,400 | 1,674,392 |

A `reg_axis_full` block has three attention sublayers (variable attention,
context read, query read) against six for `reg_axis`. A two-stage layer has
two more d × d projections than a standard attention layer and no stage-2
value projection.

## Comparison

| | r1 | layerwise | extra_node | reg_axis | reg_axis_full |
| --- | --- | --- | --- | --- | --- |
| intervention info | flag embedding at input | flag at input + re-injected before every attention | extra META node per row | flag at input + re-injected; regime grouping | flag at input + re-injected; regime grouping |
| context-row mixing | all rows, one softmax | all rows, one softmax | all rows, one softmax | within regime; across regimes via memories | every regime's rows, one softmax per regime, then a softmax over regimes |
| query reads | context rows | context rows | context rows | R·K memories per variable | all context rows, per regime, then over regimes |
| compression of a regime | none | none | none | K memory tokens per variable, query-independent | none (the reader's own summary of each regime) |
| extra tokens | none | none | one node per row | none (memories are internal) | none |
| set-size info | implicit | implicit | implicit | count feature on memories | count feature on messages |
| attention cost per block | Nc² + Nq·Nc | Nc² + Nq·Nc | Nc² + Nq·Nc | Nc² (masked) + R·K·Nc + (Nc + Nq)·R·K | (Nc + Nq)·R·N |
| zero-init equivalence | = upstream | = r1 | no | no | no |
| warm start | `--init_from_upstream` | `--init_from_r1` / upstream | upstream (not exact) | none | none |

## Limitations

- Stage 1 pads every regime to the largest regime of the batch, so its cost
  is R·N / Nc times that of attention over the real rows; with very unequal
  regime sizes this approaches a factor R. `--same_numregimes_per_batch`
  removes the padding of absent regimes only. Packed variable-length attention
  (nested tensors, or bucketing by regime size) would remove the rest.
- The stage-1 call has 5-D inputs, for which
  `F.scaled_dot_product_attention` uses its generic path on CPU and GPU: the
  [B, R, D·H, L, N] attention weights are materialised (no fused kernel). The
  messages [B, D, L, R, d] are materialised too, which a standard attention
  layer does not need.
- Context is re-encoded once per query group (the role encoding depends on
  (T, Y)), so nothing can be cached across questions.
- The bitmask regime key limits D to 62 (`group_regimes`).
- `context_states` and the flag probe do not support this encoding (see
  above).
