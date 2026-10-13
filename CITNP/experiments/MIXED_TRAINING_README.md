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
