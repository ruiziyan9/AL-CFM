# Mixed-regime task generator

`mixed_regime_generator.py` generates synthetic causal tasks for the
mixed-context model. Each task is one SCM (graph and mechanisms) plus one
random design. The context mixes an optional observational regime with up to
`k_max` single-node interventional regimes from that SCM. Each query row has
its own target (one node, or none) and its own do-value. Every non-intervened
cell of a query row is a prediction target. Tasks are iid; nothing is chosen
adaptively.

The original MACE-TNP generators are unchanged. This module only imports from
them.

## Guarantees

- **One mechanism call per node per task.** Each mechanism runs once, on the
  hidden reference rows, all context rows and all query rows stacked together.
  The repo's mechanisms are not fixed functions (GPs draw a fresh joint sample,
  and `LinearNonIdentifiable` resamples its weights on every call), so this is
  what makes all regimes come from the same SCM.
- **Reference standardisation** (`normalise=True`). Each variable is
  standardised with statistics from `n_ref` hidden observational rows, before
  its children are computed. The rows are dropped afterwards. Do-values are
  drawn in this standardised space. No other normalisation is applied.
- **Reproducibility.** Every task is identified by a key (see
  [Seeding](#seeding)). `generate_task(cfg, key)` seeds Python `random` (used
  by igraph), `np.random` and `torch` from it. A task that is non-finite or
  has any value above `max_abs_value` is regenerated from `key + (attempt,)`.
  `meta` records `seed_key`, `used_key`, `resample_attempts` and `max_abs`.
- **Random variable order.** Variables are randomly permuted, and the returned
  `graph`, `regime_targets` and query targets use the permuted indices. Context
  rows are shuffled, so regimes are not contiguous.

## Regimes vs prediction targets

`Task.regime_targets` holds the **intervention** target set of each context
regime (permuted indices); `()` is the observational regime. `ctx_regime[i]`
indexes into it, so context row i was generated under do(X_t) for every t in
`regime_targets[ctx_regime[i]]`, with its own do-value per target. Example:
`regime_targets = [(), (2,)]` and `ctx_regime = [1, 0]` means row 0 has X_2 set
by intervention and row 1 is observational. Prediction targets are unrelated
to regimes: they are the query cells `qry_tgt = ~qry_int`. Both
`regime_targets` and `ctx_regime` are not passed to the model in the flat
design; the model sees the same information through `ctx_int`.

## Config (`MixedRegimeConfig`)

| field | default | meaning |
|---|---|---|
| `function_generator` | — | registry key (`linear_iw`, `linear_nig`, `linear_nig_legacy`, `linearnonidentifiable`, `linearfixedstd`, `neuralnet`, `gp`, …) |
| `num_variables` | — | list of D values to sample from |
| `graph_type` | — | list of `ER` / `ER_CC` / `SF` / `BP` / `FC` (`ER_CC`: ER conditioned on containing a common cause, i.e. a node with ≥ 2 children; needs ≥ 2 edges) |
| `graph_degrees` | — | expected **number of edges** (`s0` of `generate_synthetic_dag`), as a list or a dict keyed by D (same convention as the original repo). ER is capped at D(D−1)/2 |
| `normalise` | `True` | reference standardisation; set `False` for oracle-matched families |
| `weight_scale` | 1.0 | edge-weight scale η of the linear-Gaussian families (`linear_iw` requires 1): w \| s² ~ N(0, η s²); the exact oracle must use the same η |
| `weight_mean` | 0.0 | signed edge-weight mean m of the linear-Gaussian families (`linear_iw` requires 0): w \| s² ~ N(s m, η s²) with sign s = ±1 w.p. ½ per edge, so weights sit away from zero; 0 is the original prior. The exact oracle (`CITNP.evaluation.oracle.LinGaussOracle(weight_mean=m)`) stays exact by summing over the 2^p sign patterns of each node's p parents |
| `alpha0` | None | root IG shape α₀ of `linear_iw` / `linear_nig` (default 2.5; non-roots get α₀ + \|pa\|/2); not allowed for the legacy keys |
| `motif_probs` | None | class probabilities of `graph_type: [MOTIF3]` (D = 3): keys `empty`, `single`, `chain`, `fork`, `collider`, `complete`, non-negative, summing to 1; the graph is uniform within its class (labelled counts 1, 6, 6, 3, 3, 6). Required for MOTIF3 and invalid otherwise; the oracle uses the exact `evaluation.tasks.motif_graph_prior` |
| `n_ref` | 256 | hidden reference rows |
| `p_obs` | 0.85 | probability the context includes an observational regime |
| `n_obs_min` | 10 | minimum observational rows; if > 0 the observational regime is always present (`p_obs` is then unused) |
| `k_max` | 4 | max number of interventional regimes (also capped at D) |
| `p_geom` | 0.5 | P(K=k) ∝ (1−p_geom)^k on {0..min(k_max, D)} |
| `p_k0` | None | P(K = 0), the share of observation-only contexts; None keeps the truncated geometric over {0..k_cap}, otherwise K ≥ 1 follows its shape |
| `n_ctx_range` | (64, 1024) | total context rows, uniform inclusive |
| `dirichlet_alpha_range` | (0.3, 3.0) | log-uniform α of the Dirichlet over regime sizes |
| `do_scale_range` | (0.5, 3.0) | std of Gaussian do-values (standardised space) |
| `p_wide_do` | 0.1 | per task: use Uniform(−w, w) do-values instead of Gaussian |
| `wide_do_range` | (3.0, 6.0) | w ~ U(range) |
| `n_qry` | 128 | query rows per task |
| `qry_type_probs` | (0.45, 0.45, 0.10) | (seen target, unseen target, empty); unavailable types are renormalised away |
| `max_total_rows_gp` | 2048 | upper bound on n_ref + N_ctx + n_qry for GP families (O(N³) cost) |
| `max_resample_attempts` | 10 | retries (non-finite or above `max_abs_value`) before raising |
| `qry_mode` | `"per_row"` | `"per_row"`: each query row has its own target (seen, unseen or empty). `"grouped"`: G question groups per task (see [Grouped query mode](#grouped-query-mode)) |
| `qry_groups` | [1, 2, 3] | allowed G in grouped mode; each must divide `n_qry` and be ≤ min(`num_variables`) |
| `max_abs_value` | 1e4 | resample a task if any generated value (reference, context or query) exceeds this in absolute value; keeps the float32 cast from overflowing |

`qry_type_probs` puts only 10% of query rows on observational queries, because
interventional prediction is the main target of the model.

Example: [`configs/mixed_linear_gaussian.yaml`](configs/mixed_linear_gaussian.yaml)
(normalised NIG linear-Gaussian, D ∈ {2..5}). Its `graph_degrees={2: [0, 1]}`
makes half of all D=2 graphs empty. This must match the oracle's graph prior
once the oracle-matched family is added.

SF graphs add `round(s0 / D)` edges per new node; when that would be 0, `s0` is
raised to D (one edge per node) instead of silently giving an empty graph.
`meta["graph_type"]` and `meta["s0"]` record what was actually used (`None`
when an explicit graph is passed).


## Linear-Gaussian prior families

The per-node prior is one frozen dataclass, `CITNP.datasets.linear_gaussian_prior.LinearGaussianPrior`, read by both the generator and the exact oracle (`LinearGaussianPrior.from_config(cfg)`). Each task's `meta` carries `prior` (its fields) and `is_iw_compatible`.

### Definitions

- **Legacy prior** (current `LinearNonIdentifiable`, now `LinearNIGLegacy`):
  - roots: s² ~ IG(2.5, 0.5);
  - every non-root, whatever its number of parents: s² ~ IG(3.0, 0.5);
  - weights: w | s² ~ N(±m, η s² I).

  IG = Inverse-Gamma (shape, scale).
- **IW prior** (inverse-Wishart; registry key `linear_iw`): one inverse-Wishart prior on the whole covariance matrix, Σ ~ IW(ν, 2β·I), equivalently precision ~ Wishart(ν, (2β·I)⁻¹), with ν = 2α₀ + D − 1. The variable means are known to be zero. For any DAG, this induces a per-node prior (Geiger & Heckerman's parameter modularity): node j with |pa_j| parents has s_j² ~ IG(α₀ + |pa_j|/2, β) and w_j | s_j² ~ N(0, η s_j² I), with η = 1/(2β) and no signed mean (m = 0). Consequence: the prior is **score-equivalent**, meaning Markov-equivalent DAGs have identical marginal likelihoods on observational data.
- **Naming rule.** "NIG" (Normal–Inverse-Gamma) names the *per-node* form, which both priors share; that is why the oracle stays closed-form for both. "IW" names the *joint* prior from which the IW family's per-node NIGs are derived. Use `linear_nig*` only for per-node NIG priors that are **not** IW-compatible, and `linear_iw` for the IW family. Do not call it NIW (Normal–inverse-Wishart): there is no prior on the means, because they are fixed at zero. NIW would only be correct if intercepts or unknown means were added later.
- **MEC** = Markov equivalence class, identified by skeleton plus v-structures. For D=3: 25 DAGs in 11 MECs.
- The legacy prior equals the IW prior with α₀ = 2.5, β = 0.5, η = 1, m = 0 on every node with at most one parent. It differs for nodes with two parents: 3.0 instead of 3.5. So the legacy prior is score-equivalent for graphs with in-degree ≤ 1 (any D, e.g. D=3 chains and forks), but not for the complete DAG or colliders.

### Registry keys

| key | mechanism | prior |
|---|---|---|
| `linear_iw` | `LinearNIG` | IW-compatible only (`alpha0`, β = 0.5, η = 1, m = 0) |
| `linear_nig` | `LinearNIG` | free per-node NIG (`alpha0`, `weight_scale`, `weight_mean`); not necessarily IW-compatible |
| `linear_nig_legacy` | `LinearNIGLegacy` | legacy prior |
| `linearnonidentifiable` | `LinearNIGLegacy` | deprecated alias of `linear_nig_legacy`; emits a `DeprecationWarning` once per process |

Only the legacy keys patch `eta` / `weight_mean` onto the mechanisms after construction (so the goldens hold); the new keys receive the prior at construction.

## Graph-independence rule

Questions and designs must not depend on the true graph. `sample_design` takes only the config and D (never a graph), so which regimes the context contains, their sizes, the do-value distribution and the query treatments are independent of the DAG. A graph-dependent question or design (e.g. querying only true ancestors, or intervening on true roots) would leak information about the graph that the exact oracle does not model, so the oracle would no longer be Bayes-optimal and the model could exploit the leak. `testing/test_motif_prior.py` checks this.

## Batch contract (`collate_tasks`)

Batches are padded to the batch maxima (or `pad_D`). Padding masks follow the
PyTorch convention: `True` means padded.

| key | shape | dtype | semantics |
|---|---|---|---|
| `ctx_x` | [B, Nc, D] | float32 | context values; intervened cells hold the do-value; 0 in padding |
| `ctx_int` | [B, Nc, D] | bool | cell was intervened on; False in padding |
| `ctx_pad` | [B, Nc] | bool | True = padded row |
| `qry_x` | [B, Nq, D] | float32 | true values, including targets; 0 in padding |
| `qry_int` | [B, Nq, D] | bool | do-cells of the query |
| `qry_tgt` | [B, Nq, D] | bool | cells to predict (`~qry_int`, False in padding) |
| `qry_pad` | [B, Nq] | bool | True = padded row |
| `var_pad` | [B, D] | bool | True = padded variable |
| `graph` | [B, D, D] | int8 | `graph[i, j] = 1` iff i → j; diagnostics only |
| `qry_group` | [B, Nq] | int64 | question group of each query row in grouped mode; −1 in per-row mode and in padding |
| `ctx_regime` | [B, Nc] | int64 | regime index; −1 in padding; not passed to the model in the flat design (the model sees the same information through `ctx_int`) |
| `meta` | list[dict] | — | per-task design parameters, seed_key, query types, max_abs, device, … |

Both datasets yield **already-collated batches**, so use
`DataLoader(ds, batch_size=None)`.

- `MixedRegimeTaskDataset(cfg, batch_size, base_seed, same_variablenum_per_batch=True, iterations_per_epoch=None, split="train", fresh_each_epoch=True, same_nctx_per_batch=True)`
  generates on the fly. Task key =
  `make_key(SOURCE_STREAM, split_id, base_seed, epoch, worker_id, counter)`.
  GP and NN mechanisms move tensors to CUDA when it is available, so for those
  families use `num_workers=0` or pre-generated chunks.
- `MixedRegimeChunkDataset(chunk_dir, batch_size, shuffle=True, seed=0, same_variablenum_per_batch=True, drop_last=False, pad_D=None, num_workers=0)`
  streams chunk files, shards them across workers, shuffles through a buffer
  and buckets by the exact shape (D, N_ctx). Chunks are loaded with `torch.load(weights_only=False)`, so
  only load chunks you generated. It warns if `config.json` says the chunks
  were generated on CUDA. With `drop_last=False` it emits one partial batch per
  shape at the end of each epoch, so batch sizes vary; use `drop_last=True` for
  training if fixed batch sizes matter. With aligned shape groups (see
  [Padding-free batches](#padding-free-batches)) there are no partial batches.
  `len(ds)` is the exact number of batches per epoch under those conditions and
  raises `TypeError` otherwise, since `CausalInferenceTrainer` uses it to run
  validation on an epoch's last batch. With `same_variablenum_per_batch=False`
  (one bucket, padding allowed), pass the DataLoader's `num_workers` so `len`
  can count partial batches per worker.

## Grouped query mode

With `qry_mode: grouped`, each task carries G question groups (G drawn from
`qry_groups`). Group g is one upstream MACE-TNP question: a contiguous block of
`n_qry // G` query rows that all intervene on the same single treatment T_g,
with independent do-values per row. Treatments are distinct across groups and
chosen as "seen" (intervened on in the context) or "unseen" with weights
`qry_type_probs[0:2]`; the "empty" type is not used, because upstream's role
embedding needs a treatment. The outcome Y_g is chosen later by the trainer.

- `Task.qry_group = repeat(arange(G), n_qry // G)`; query rows are never
  shuffled, so blocks stay contiguous.
- `meta["n_groups"]`, `meta["group_treatments"]` (permuted indices) and
  `meta["group_types"]`. In per-row mode they are 0, [] and [], and
  `qry_group` is all −1.
- `generate_task(..., n_groups=G)` fixes G (not with an explicit design, and
  only in grouped mode). The context is drawn exactly as for a per-row task
  with the same key; G is drawn after the context design, before the query
  targets.
- Batches have one G: the streaming dataset draws it per batch after D and
  N_ctx; the chunk writer draws it per shape group after D and N_ctx; the
  chunk dataset buckets grouped tasks by (D, N_ctx, G).
- Grouped mode is purely additive: per-row generation, batches and chunks are
  byte-identical to before (regression tests), and chunks written before this
  mode existed still load (`qry_group` defaults to −1).

## Padding-free batches

The current encoder cannot mask padding: it has no row-padding masks, and its
variable-padding mask is a TODO. Every batch must therefore have one N_ctx,
one N_qry and one D across its tasks. N_qry is always `cfg.n_qry`.

- **Streaming:** `same_variablenum_per_batch=True` and `same_nctx_per_batch=True`
  (both defaults) draw one D and one N_ctx ~ U{`n_ctx_range`} per batch, from
  the batch's `SOURCE_STREAM_D` key (D first, then N_ctx). Each task's
  marginal prior is unchanged; only which tasks share a batch changes.
- **Chunks:** `--shape_group_size G` (default 256) makes consecutive groups of
  G tasks share one (D, N_ctx), drawn from the group's own key
  `make_key(SOURCE_CHUNK, split_id, base_seed, global_task_index // G, 1, 0)`.
  **`chunk_size` and `num_tasks` must be multiples of G** (the writer enforces
  this): chunk files are sharded across DataLoader workers file by file, so a
  group straddling two files would be split across workers and end in partial
  batches. **G should be a multiple of the training batch size**, so each group
  splits into whole batches; then `len(MixedRegimeChunkDataset)` is exact
  (otherwise `MixedRegimeChunkDataset` warns and `len` raises).
  `--shape_group_size 0` turns grouping off (per-task shapes, which then need
  padding). `MixedRegimeChunkDataset` buckets by the exact
  shape (D, N_ctx). Keep `pad_D=None`: it reintroduces variable padding.
- `generate_task(cfg, key, num_variables=D, n_ctx=N)` fixes both by hand
  (`n_ctx` cannot be combined with an explicit `design`). Without `n_ctx`,
  generation is byte-identical to earlier versions.
- `MixedRegimeConfig` requires `n_ctx_range[0] >= min(k_max, max(num_variables)) + 1`,
  so the floor of N_ctx at the number of regimes never changes a fixed N_ctx.

## Seeding

Every task is identified by a fixed-length key
`(source, split_id, base_seed, i1, i2, i3)` built with `make_key`. All RNG
states are derived from it with `np.random.SeedSequence`, which hashes the
whole key, so any two different keys give unrelated tasks. (The old additive
seeds overlapped: base seeds 0 and 1 shared almost all tasks.)

| source | key | used by |
|---|---|---|
| `SOURCE_ADHOC` (0) | `(0, 0, base_seed, i, 0, 0)` | tests, `python -m CITNP.prior_diagnostics` (all commands); `generate_task(cfg, i)` with an int |
| `SOURCE_CHUNK` (1) | `(1, split_id, base_seed, global_task_index, 0, 0)` | chunk writer |
| `SOURCE_STREAM` (2) | `(2, split_id, base_seed, epoch, worker_id, counter)` | `MixedRegimeTaskDataset` |
| `SOURCE_STREAM_D` (3) | `(3, split_id, base_seed, epoch, worker_id, batch_idx)` | per-batch D and N_ctx in `MixedRegimeTaskDataset` |

The chunk writer's shape groups use `(1, split_id, base_seed, group_index, 1, 0)`;
`i2 = 1` keeps them apart from task keys.

`SPLIT_IDS = {"train": 0, "val": 1, "test": 2}`. **Train, validation and test
must use different splits** (`--split` for chunks, `split=` for streams); then
they never share tasks, even with the same base seed. Before training on chunk
directories, check them:

```python
from CITNP.datasets.mixed_regime_generator import assert_disjoint_splits

assert_disjoint_splits("data/mixed_lg/train", "data/mixed_lg/val")
```

It raises if two directories have the same `split_id` and `base_seed`.

**Every dataset mixed into training must use a distinct `base_seed`** (for
example, one per mechanism family). Task keys do not include the config, so two
datasets with the same split and base seed draw identical random streams.

## Training with CausalInferenceTrainer

CITNP's custom trainer calls callbacks from `CITNP/trainer/callbacks.py` by
name, calls `on_epoch_begin(trainer, epoch)` before it creates the epoch's
training iterator, and sets `persistent_workers=True` whenever
`num_workers > 0`.

`MixedRegimeTaskDataset(..., fresh_each_epoch=True)` (the default) keeps its
key counters on the object, so fresh tasks per epoch work in all three
DataLoader setups:

- **persistent workers** (what the trainer uses when `num_workers > 0`): each
  worker's copy keeps its counters, so later epochs continue with new keys;
- **`num_workers=0`**: a single object, counters keep growing;
- **non-persistent workers**: each epoch starts from a fresh copy (counters
  reset), so `set_epoch(epoch)` must be called each epoch. `SetEpochCallback`
  in `epoch_callback.py` does this from `on_epoch_begin`.

```python
from CITNP.datasets.epoch_callback import SetEpochCallback

train_ds = MixedRegimeTaskDataset(cfg, 32, base_seed=0, split="train")
val_ds = MixedRegimeTaskDataset(
    cfg, 32, base_seed=0, split="val", iterations_per_epoch=50, fresh_each_epoch=False
)
callbacks = [SetEpochCallback([train_ds])]  # training datasets only
```

**Validation and test** must see the same tasks every time: use
`fresh_each_epoch=False` for streams, and `shuffle=False` for chunk datasets.
The trainer runs validation many times per epoch, and reshuffling changes the
batch composition, and with it the mean of per-batch losses.

`MixedRegimeChunkDataset` advances its own epoch in `__iter__`, which persists
in persistent workers; with non-persistent workers, pass it to
`SetEpochCallback` too.

## Generating chunks (Oxford ARC / SLURM)

```bash
python -m CITNP.datasets.create_mixed_regime_chunks \
    --config CITNP/datasets/configs/mixed_linear_gaussian.yaml \
    --num_tasks 102400 --chunk_size 1024 \
    --out_dir $DATA/mixed_lg/train --split train --base_seed 0 \
    --shape_group_size 256 \
    [--chunk_index $SLURM_ARRAY_TASK_ID] [--allow_cuda]
```

`chunk_size` and `num_tasks` must be multiples of `shape_group_size` (the
writer rejects anything else), and `shape_group_size` should be a multiple of
the training batch size. The example gives 100 chunks of 1024 tasks, four shape
groups per chunk, and fits batch sizes 8, 16, 32, 64, 128 and 256.

Run it once per split (`--split train`, `--split val`, `--split test`), each
into its own directory. Each chunk `chunk_{k:05d}.pt` holds a list of unpadded
task dicts (`dataclasses.asdict(Task)`). Task key =
`make_key(SOURCE_CHUNK, split_id, base_seed, global_task_index)`, and shape
groups are keyed by their global group index, so running all chunks in one
process gives the same output as one chunk per array job.
A full SLURM array script is in the docstring at the top of
`create_mixed_regime_chunks.py` (`--array=0-99` for the numbers above, i.e.
`num_tasks / chunk_size - 1`).

**Device.** The writer runs on CPU by default: torch's CPU and CUDA random
streams differ, so the same key would give different GP/NN tasks on GPU and CPU
nodes. `--allow_cuda` lets GP/NN mechanisms use the GPU; tasks are then
reproducible only on the same device type. The writer refuses to run if CUDA is
visible without `--allow_cuda`. `config.json` records the config, split,
`split_id`, `base_seed`, `shape_group_size`, git commit, `device`, `torch_version`,
`numpy_version` and `python_version`; each task's `meta["device"]` records the
device its mechanisms ran on (`"cpu"` for the CPU-only families
`CPU_ONLY_FAMILIES`: `linearfixedstd`, `linearnonidentifiable`, `uniformlinear`,
`sinusoid`, `resnet`).

## Diagnostic options

Two optional arguments exist for diagnostics (used by `CITNP/prior_diagnostics`).
Both are off by default, and generation with the defaults is byte-identical to
before (same random stream).

- `Design.regime_do_values: list[float | None] | None = None`: one entry per
  context regime. A float fixes every intervened cell of that regime to that
  value instead of a draw from `do_dist`; `None` keeps the usual draws. The
  draws still happen and are then overwritten, so the random stream is unchanged.
- `generate_task(..., permute=False)`: skips the random variable permutation, so a task
  generated from an explicit `graph` / `design` keeps their variable labels.
  The permutation is still drawn, so the random stream is unchanged.

## Diagnostics and tests

Prior visualisations live in [`CITNP/prior_diagnostics`](../prior_diagnostics/README.md):

```bash
python -m CITNP.prior_diagnostics summary --config <cfg> --num_tasks 2000 --out prior.png
pytest testing/test_mixed_regime_generator.py testing/test_prior_diagnostics.py
```

## Not done yet

- **Regenerate existing chunks.** Chunks written before key-based seeding and
  the `max_abs_value` check use the old seeds and acceptance rule; generate them
  again with `--split`. `assert_disjoint_splits` rejects old directories (no
  `split_id` in `config.json`). Chunks whose `chunk_size` or `num_tasks` is not
  a multiple of `shape_group_size` split groups across workers; regenerate them
  with aligned sizes. Chunks written without `--shape_group_size`
  mix context sizes within a batch and would need padding, so regenerate those
  too.

- The D=2 family matched to the `Belief` oracle, and its moment test (spec
  §8 / §10.2). Deferred until the oracle's hyperparameters and graph prior are
  confirmed.
- Analytic interventional moments for linear-Gaussian queries (optional).
