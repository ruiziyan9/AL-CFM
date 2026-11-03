# `reg_axis`: regime-aware intervention encoding

`reg_axis` is the fourth intervention encoding of the mixed-context MACE-TNP
model, next to `r1`, `layerwise` and `extra_node`
([`mixed_context_model.py`](../mixed_context_model.py)). Each experimental
regime is treated as its own unit. Rows exchange information only with rows
of their own regime. Each regime is then compressed into K memory tokens per
variable. Regimes exchange information through those memories, and query rows
read the fused memories. Everything else is reused unchanged: the six role
MLPs, the target masking, the prediction head, the losses, the trainer, the
data, scale conditioning and evaluation.

Notation: B = tasks × query groups (the flattened batch), Nc context rows,
Nq query rows, D variables, d = `d_model`, H = `nhead`, K memory tokens,
R = R_max regime slots in the batch.

## Motivation

The context is a union of experiments: an observational sample and samples
from do(X_S = ·) for several target sets S. Under the SCM the likelihood
factorises over regimes, with each experiment informative about the
mechanisms it leaves intact. `reg_axis` builds this structure into the
encoder. Each experiment is processed on its own first (within-regime sample
attention, then pooling to a summary), and only then are the experiments
combined (attention across the regime summaries). The other encodings let
every context row attend to every other row and leave the model to work out
which rows belong together from a per-cell flag.

## Regimes, keys and slots ([`grouping.py`](grouping.py))

A regime is defined by its **intervention-target set** only, never by the
intervention values. For row i with target mask a_i ∈ {0,1}^D (row i of
`context_int_mask`):

- key: `key_i = Σ_j a_ij 2^j` (int64, so D ≤ 62);
- slots: within each task, the sorted distinct keys present map to slots
  0..R_b−1. R = max_b R_b; tasks with fewer regimes have *absent* slots.

Keys and slots are bookkeeping. They are never embedded, and there are no
regime identifiers or positional embeddings, so the model is invariant to
row order and to slot order. `group_regimes` returns a frozen `RegimeGrouping`:

| field | shape | meaning |
| --- | --- | --- |
| `slot` | [B, Nc] long | slot of each row, −1 on padded rows |
| `same_regime` | [B, Nc, Nc] bool | sample-attention mask (True = attend) |
| `slot_member` | [B, R, Nc] bool | row i belongs to slot r |
| `slot_present` | [B, R] bool | slot r exists in task b |
| `n_r` | [B, R] long | real rows per slot (0 if absent) |

The optional `context_regime` labels ([B, Nc], −1 = padding) are used only as
a consistency check. Every label must cover a single target set. Several
labels may share one set, and they then merge into one slot. Optional
`context_row_valid` keeps a code path for future row padding. Padded rows
belong to no slot and are never attended to by real rows; each one attends
only to itself, so its values stay finite. The trainer sends no padding
today, and the model does not use this argument yet.

## Input encoding ([`encoder.py`](encoder.py), [`encoding.py`](encoding.py))

Per cell token, before the first block:

```text
h0 = role MLP embedding                                  (upstream embed())
   + flag * flag_embed(x)    on intervened context cells (target_flag injected)
   + ScaleConditioning(stats_j) on every row of column j (scale_conditioning stats)
```

- **Role MLPs.** `RegimeAxisEncoder` subclasses upstream `CausalInfEncoder`,
  so the six MLPs keep their names (`encoder.context_variable.*`, …). The
  inherited transformer stack (`self.encoder`) is deleted and replaced by
  `self.blocks`.
- **Treatment role vs intervention target.** The *treatment role* is the
  column of the query's T; the `*_intervention` role MLPs mark it in context
  and query rows, as upstream. An *intervention target* is a context cell set
  by an experiment (`context_int_mask`). The two are independent: the role
  depends on the question (T, Y), the target on the row's experiment.
- **Query masking.** The model calls upstream `_mask_target_data` first, so
  query rows carry only the treatment value t. Hidden outcomes never reach the
  encoder.
- **Flag** (`reg_axis_target_flag: injected`, the default). This mirrors the
  functional form of `layerwise`, with new parameters under `encoder.flag.`:
  `flag_embed` (MLP 1→d, last layer zero-initialised); `flag_code`
  (MLP 1→d, default init); `code = flag * flag_code(x)`, exactly zero on
  non-intervened cells and on query rows; and before steps 1 and 2 of block l,
  `hc += inject_var[l](code)` and `hc += inject_sample[l](code)`
  (`Linear(d, d, bias=False)`, zero-initialised). Query rows get no flag.
  With `none` there are no flag parameters, and regimes are known only
  through the grouping (a structure-only ablation).
- **No extra node or token.** The flag is a per-cell feature and the count is
  a per-memory feature (below). Neither adds a token.

## One block ([`block.py`](block.py))

Context tokens `hc` [B, Nc, D, d]; query tokens `hq` [B, Nq, D, d].

| step | operation | attention tensor | mask |
| --- | --- | --- | --- |
| 0 | `hc += inject_var[l](code)` | — | — |
| 1 | variable attention, context and query rows, shared weights | [B·(Nc+Nq), D, d] | none |
| 0' | `hc += inject_sample[l](code)` | — | — |
| 2 | within-regime sample attention per variable | [B, D, Nc, d] | `same_regime` → [B, 1, Nc, Nc] |
| 3 | PMA: R·K queries per (task, variable) read that task's rows | q [B, D, R·K, d], kv [B, D, Nc, d] | `slot_member` ×K + empty-slot guard |
| 3' | `+ Linear(1, d)(log1p(n_r))` on every memory of slot r | [B, D, R, K, d] | — |
| 4 | regime attention per variable over all R·K memories | [B·D, R·K, d] | key padding from `slot_present` |
| 5 | context feedback: cells of variable j read j's memories (not in the last block) | q [B·D, Nc, d], kv [B·D, R·K, d] | key padding |
| 6 | query read: query cells of variable j read j's memories | q [B·D, Nq, d], kv [B·D, R·K, d] | key padding |

```text
          context hc [B,Nc,D,d]                     query hq [B,Nq,D,d]
                 │ (+inject_var)                           │
                 └──────────► 1. variable attention ◄──────┘   (shared, per row)
                 │ (+inject_sample)                        │
   2. sample attention within each regime                  │
                 │                                         │
   3. PMA → memories [B,D,R,K,d] (+count)                  │
                 │                                         │
   4. regime attention over R·K memories per variable      │
          │                   │                            │
   5. feedback: hc reads      └──────────────► 6. query read: hq reads memories
      memories (not in last block)                         │
          ▼                                                ▼
        next block                                     next block / head
```

After the last block the query tokens go through upstream
`_extract_outcome_representation(rep, outcome_index, num_trgt=Nq)` (the
outcome column of every query row) and the upstream MoG predictor.

Every sublayer is pre-norm and residual, with a GELU FFN of width
`dim_feedforward` and dropout `dropout`. Steps 1, 4, 5 and 6 reuse upstream
`MultiHeadSelfAttentionLayer` / `MultiHeadCrossAttentionLayer`, whose masks
are small there. Steps 2 and 3 use `MaskedSetAttentionLayer`
([`attention.py`](attention.py)). It has the same structure, but its boolean
mask (True = attend) is passed to SDPA once per task and broadcast over
variables and heads. Upstream layers copy a 3-D mask once per head, which
for step 2 would be B·D·H·Nc² booleans, about 1.6 GB at B=64, D=3, H=8,
Nc=1024. Internally the D and H axes are merged ([B, D·H, L, d/H]), so SDPA
sees 4-D inputs with a [B, 1, L, S] mask.

The last block has no feedback sublayer. Its output could never reach a
prediction, and an unused module would leave parameters without gradients,
which breaks DistributedDataParallel without `find_unused_parameters`. This
mirrors `layerwise`, which has no node injection in its last block.

## Memories, PMA and the empty-slot guard ([`pooling.py`](pooling.py))

`q_mem` ([K, d], one set per block, shared across regimes and variables) is
repeated for the R slots. The queries of slot r attend to the rows of slot r.
The output is `q_mem + attention`, followed by the FFN.

**Empty-slot guard.** An absent slot has no member rows, so its mask row would
be all False. A plain softmax then returns NaN, which poisons the backward
pass even if the value is zeroed afterwards (`torch.where` does not help).
PyTorch ≥ 2.5 SDPA happens to return zeros for such rows, but the model does
not rely on this. Absent slots may read every row
(`pool_mask = slot_member | ~slot_present`), their memories are multiplied by
`slot_present`, and steps 4–6 mask them out as keys. Every task has at least
one regime, so every attention row in steps 4–6 has an allowed key.

**Count feature** (`reg_axis_count_feature: true`). PMA is a softmax average
and cannot see the size of a set. Without the count, a 1-row and a 300-row
regime with the same empirical distribution give identical memories, and
duplicating every row of a regime changes nothing. `Linear(1, d)(log1p(n_r))`
is added to every memory of slot r (one Linear per block). With `false` the
Linear is not built.

## Missing regimes

A task contains whichever regimes the generator drew: only observational,
observational plus some experiments, or experiments without observational
data. Absent slots contribute no keys anywhere, so a task with R_b regimes
always has exactly R_b·K present memories per variable. No regime has a fixed
position, so nothing needs to be imputed.

## Query conditioning and no leakage

- Query rows carry only t (upstream masking). They are mixed with other
  variables of the same row (step 1) and read memories (step 6).
- Query rows never attend to each other or to context rows directly, and
  context rows and memories never read query rows. Predictions therefore do
  not depend on the hidden query values, and adding a query row leaves the
  other predictions unchanged (`test_query_leakage`).
- Context is re-encoded once per query group, because the role embedding
  depends on (T, Y).

## Configuration

| key (model YAML / `CausalInfModelConfig`) | default | meaning |
| --- | --- | --- |
| `intervention_encoding: reg_axis` | — | selects `RegimeAxisMixedContextMoGModel` |
| `reg_axis_num_blocks` | `null` → `num_layers_encoder // 2` | number of blocks |
| `reg_axis_num_memory_tokens` | 4 | K |
| `reg_axis_target_flag` | `injected` | `injected` or `none` |
| `reg_axis_count_feature` | `true` | count feature on memories |

`sample_attn_mode` is ignored. `linear_attention: true` is rejected, because
the regime masks need softmax attention. `variable_mask` is not supported.
`scale_conditioning` works exactly as for the other encodings. The
`reg_axis_*` keys are rejected for other encodings, and
`--init_from_upstream` is rejected for `reg_axis`, whose transformer
parameters have no upstream counterpart. Configs:
[`smoke_d3_model_reg_axis.yaml`](../../experiments/configs/smoke_d3_model_reg_axis.yaml)
and
[`oracle_d3_model_m_reg_axis_stats.yaml`](../../experiments/configs/oracle_d3_model_m_reg_axis_stats.yaml),
which is identical to `oracle_d3_model_m_layerwise_stats.yaml` apart from the
encoding.

Smoke run (CPU):

```bash
python -m CITNP.experiments.mixed_train \
    --data_config CITNP/experiments/configs/smoke_d3.yaml \
    --model_config CITNP/experiments/configs/smoke_d3_model_reg_axis.yaml \
    --batch_size 16 --iterations_per_epoch 50 --epochs 3 \
    --num_workers 0 --seed 0 --device cpu --no_wandb --run_name smoke_reg_axis
```

Full training: the same command as for the other encodings
([`MIXED_TRAINING_README.md`](../../experiments/MIXED_TRAINING_README.md)),
with `--model_config CITNP/experiments/configs/oracle_d3_model_m_reg_axis_stats.yaml`.
`model_api.load_model` reads the `reg_axis_*` keys from the run's
`resolved_model_config.yaml`.

**`zero_flags` caveat.** With `zero_flags=True` in `predict_mixture` /
`context_states`, `context_int_mask` is all False. For `reg_axis` this also
removes the regime grouping, since every row then falls into one
observational regime. The ablation therefore removes both the flag and the
regime structure, which is a stronger change than for `r1` / `layerwise`
(flag only) or `extra_node` (metadata only).

## Comparison

| | r1 | layerwise | extra_node | reg_axis |
| --- | --- | --- | --- | --- |
| intervention info | flag embedding at input | flag at input + re-injected before every attention | extra META node per row | flag at input + re-injected; regime grouping |
| context-row mixing | all rows | all rows | all rows | within regime; across regimes via memories |
| query reads | context rows (MHCA/MHSA) | context rows | context rows | R·K memories per variable |
| extra tokens | none | none | one node per row | none (memories are internal) |
| set-size info | implicit (attention over rows) | implicit | implicit | explicit count feature |
| zero-init equivalence | = upstream | = r1 | no | no |
| warm start | `--init_from_upstream` | `--init_from_r1` / upstream | upstream (not exact) | none |

Parameter counts (MoG models, D = 3; reg_axis with its default
`reg_axis_num_blocks` = `num_layers_encoder // 2` and K = 4):

| config | r1 | layerwise | extra_node | reg_axis |
| --- | --- | --- | --- | --- |
| `smoke_d3_model*.yaml` (d 64, 4 layers) | 238,601 | 255,177 | 234,569 | 426,441 |
| `oracle_d3_model_m*.yaml` (d 128, 6 layers) | 1,342,360 | 1,441,048 | 1,326,104 | 2,516,504 |
| `oracle_d3_model_m*_stats.yaml` | 1,359,256 | 1,457,944 | 1,343,000 | 2,533,400 |

A reg_axis block has six attention sublayers (five in the last block), against
three for an upstream MHCA sample/node pair, so at equal `num_layers_encoder`
reg_axis has more parameters. Use `reg_axis_num_blocks` to match budgets.

## Limitations

- Context is re-encoded once per query group (the role encoding depends on
  (T, Y)), so memories cannot be cached across questions.
- Sample attention still costs Nc² per variable, even though cross-regime
  blocks are masked. A later optimisation could sort rows by regime in a
  reg_axis-only collate step and use packed variable-length attention (nested
  tensors or `flex_attention`). The model must keep accepting arbitrary row
  order.
- The bitmask key limits D to 62.
- On CPU the SDPA math path materialises the [B, D·H, Nc, Nc] attention
  weights, as upstream does.
