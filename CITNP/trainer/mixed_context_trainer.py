import math
from pathlib import Path
"""
Trainer for the mixed-context model (design R1).

Batches come from MixedRegimeTaskDataset / MixedRegimeChunkDataset in grouped
query mode: already collated and padding-free, one G (number of question
groups) per batch. `prepare_mixed_batch` turns a batch of B tasks into B * G
upstream MACE-TNP questions: it reads each group's treatment T from qry_int,
draws an outcome Y != T, normalises per task, and duplicates the task's
context for every group.

MixedContextTrainer overrides only what differs from CausalInferenceTrainer:
the loaders (no collate_fn), _prepare_batch, _forward_pass (passes
context_int_mask), validate (marks batches as evaluation, for deterministic Y)
and test_epoch (reports losses by stratum).
"""

import hashlib
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch as th
from tqdm import tqdm
import math
from pathlib import Path

from CITNP.trainer.causalinferencetrainer import CausalInferenceTrainer
from CITNP.utils.utils import get_causal_direction


def _train_generator(seed: int, global_step: int) -> th.Generator:
    """Generator for the training outcomes of one step."""
    state = np.random.SeedSequence([seed, global_step]).generate_state(1)
    return th.Generator().manual_seed(int(state[0]))


def _eval_outcome_draw(seed_key: Any, group: int, num_choices: int) -> int:
    """Deterministic draw in [0, num_choices) for one (task, group)."""
    digest = hashlib.sha256(f"{seed_key}-{group}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % num_choices


def _normalisation_stats(
    ctx_x: th.Tensor, ctx_int: th.Tensor, min_rows: int = 10
) -> tuple[th.Tensor, th.Tensor]:
    """Per-task, per-variable (mean, std) [B, D] from observational context rows
    only (rows with no intervened variable).

    Requires at least `min_rows` observational rows per task (guaranteed by the
    generator's n_obs_min); raises otherwise instead of silently mis-scaling.
    """
    x = ctx_x.double()                                     # [B, Nc, D]
    obs_row = ~ctx_int.any(dim=-1, keepdim=True)           # [B, Nc, 1]
    m = obs_row.double().expand_as(x)                      # same rows for every variable
    n = m.sum(1)                                           # [B, D], equal across D
    if (n < min_rows).any():
        raise ValueError(
            f"{int((n[:, 0] < min_rows).sum())} task(s) have fewer than {min_rows} "
            f"observational context rows (min {int(n.min())}); set n_obs_min in the data config"
        )
    mean = (x * m).sum(1) / n
    std = ((((x - mean[:, None]) ** 2) * m).sum(1) / n).sqrt()
    return mean.float(), std.clamp(min=1e-6).float()


def prepare_mixed_batch(
    batch: Dict[str, Any],
    dtype: th.dtype,
    training: bool,
    *,
    device: th.device | str = "cpu",
    seed: int = 0,
    global_step: int = 0,
    normalise: bool = True,
) -> Dict[str, Any]:
    """Turn a grouped batch of B tasks into B * G upstream questions.

    Args:
    ----------
    batch : dict
        Output of collate_tasks (grouped qry_mode, padding-free).

    dtype : th.dtype
        Float dtype for the model inputs; bools and longs keep their dtype.

    training : bool
        True: outcomes Y are drawn from a generator seeded by
        (seed, global_step), so they differ between steps. False
        (validation / test): Y is a hash of (seed_key, group), identical on
        every call.

    device, seed, global_step, normalise :
        Target device, trainer seed, current step, and whether to standardise
        each task with the (mean, std) of its non-intervened context cells.

    Returns:
    ----------
    dict with "context" [B*G, Nc, D, 1], "target" [B*G, Nq_g, D, 1],
    "intervention_index" / "outcome_index" [B*G] (long), "context_int_mask"
    [B*G, Nc, D] (bool), "causal_graphs" [B*G, D, D], "group_types" (list,
    flattened in (b, g) order), "meta", "norm" = (mean, std) [B, D], "B", "G".
    """
    ctx_x, ctx_int = batch["ctx_x"], batch["ctx_int"]
    qry_x, qry_int, qry_group = (
        batch["qry_x"],
        batch["qry_int"],
        batch["qry_group"],
    )
    meta = batch["meta"]
    B, Nc, D = ctx_x.shape
    Nq = qry_x.shape[1]

    # 1. Guards.
    if batch["ctx_pad"].any() or batch["qry_pad"].any() or batch["var_pad"].any():
        raise ValueError(
            "padding-free batches required (same D, N_ctx and G per batch)"
        )
    if (qry_group < 0).any():
        raise ValueError(
            "grouped query mode required: qry_group is -1 (per-row batch); set "
            "qry_mode='grouped' in the data config"
        )
    G = int(qry_group.max()) + 1
    if Nq % G != 0:
        raise ValueError(f"Nq={Nq} is not divisible by G={G}")
    Nq_g = Nq // G
    expected = th.arange(G).repeat_interleave(Nq_g)
    if not (qry_group == expected[None]).all():
        raise ValueError(
            "qry_group must be contiguous equal blocks 0..G-1 in every task"
        )

    # 2. Queries per group.
    qx = qry_x.view(B, G, Nq_g, D)
    qi = qry_int.view(B, G, Nq_g, D)

    # 3. Treatment per (task, group).
    if not (qi.sum(-1) == 1).all():
        raise ValueError(
            "every grouped query row must intervene on exactly one variable"
        )
    if not (qi == qi[:, :, :1]).all():
        raise ValueError("all query rows of a group must share one treatment")
    T = qi[:, :, 0].float().argmax(-1)  # [B, G]
    T_sorted = T.sort(dim=1).values
    if G > 1 and not (T_sorted[:, 1:] != T_sorted[:, :-1]).all():
        raise ValueError("treatments must be distinct across a task's groups")
    for b in range(B):
        if "group_treatments" in meta[b] and meta[b]["group_treatments"]:
            assert T[b].tolist() == meta[b]["group_treatments"], (
                b,
                T[b],
                meta[b],
            )

    # 4. Outcome Y uniform over {0..D-1} \ {T}.
    if training:
        r = th.randint(
            0, D - 1, (B, G), generator=_train_generator(seed, global_step)
        )
    else:
        r = th.tensor(
            [
                [
                    _eval_outcome_draw(meta[b]["seed_key"], g, D - 1)
                    for g in range(G)
                ]
                for b in range(B)
            ],
            dtype=th.long,
        )
    Y = r + (r >= T).long()  # skips T

    # 5. Per-task normalisation, shared by the task's groups.
    if normalise:
        mean, std = _normalisation_stats(ctx_x, ctx_int)
    else:
        mean, std = th.zeros(B, D), th.ones(B, D)
    ctx_n = (ctx_x - mean[:, None]) / std[:, None]
    qx_n = (qx - mean[:, None, None]) / std[:, None, None]

    # 6. Flatten (task, group) into the batch dimension.
    ctx = ctx_n[:, None].expand(B, G, Nc, D).reshape(B * G, Nc, D, 1)
    cint = ctx_int[:, None].expand(B, G, Nc, D).reshape(B * G, Nc, D)
    tgt = qx_n.reshape(B * G, Nq_g, D, 1)
    graph_f = batch["graph"][:, None].expand(B, G, D, D).reshape(B * G, D, D)
    group_types = [
        (meta[b].get("group_types") or [None] * G)[g]
        for b in range(B)
        for g in range(G)
    ]

    # 7. Device and dtype.
    return {
        "context": ctx.to(device=device, dtype=dtype),
        "target": tgt.to(device=device, dtype=dtype),
        "intervention_index": T.reshape(B * G).to(device),
        "outcome_index": Y.reshape(B * G).to(device),
        "context_int_mask": cint.to(device),
        "causal_graphs": graph_f.float().to(device),
        "group_types": group_types,
        "meta": meta,
        "norm": (mean, std),
        "B": B,
        "G": G,
    }


class MixedContextTrainer(CausalInferenceTrainer):
    """CausalInferenceTrainer for grouped mixed-context batches.

    Extra argument `seed` seeds the training outcome draws. Datasets must
    yield collated grouped batches and define __len__ (MixedRegimeTaskDataset
    with iterations_per_epoch, or MixedRegimeChunkDataset).
    """

    def __init__(self, *args, seed: int = 0, **kwargs):
        self.seed = seed
        # True while preparing validation / test batches (deterministic Y).
        self._eval_batches = False
        super().__init__(*args, **kwargs)

    def initialise_loaders(self):
        def loader(ds):
            num_workers = self.dataconfig.num_workers
            return th.utils.data.DataLoader(
                ds,
                batch_size=None,
                collate_fn=None,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )

        self.train_loader = loader(self.train_dataset)
        self.val_loader = loader(self.validation_dataset)
        self.test_loader = loader(self.test_dataset)

    def _prepare_batch(self, batch, dtype: th.dtype) -> Dict[str, Any]:
        return prepare_mixed_batch(
            batch,
            dtype,
            training=not self._eval_batches,
            device=self.device,
            seed=self.seed,
            global_step=self.global_step,
            normalise=self.normalise,
        )

    def _forward_pass(
        self, batch: Dict[str, Any], test: Optional[bool] = False
    ) -> Dict[str, Any]:
        model_output = self.model(
            context_data=batch["context"],
            target_data=batch["target"],
            intervention_index=batch["intervention_index"],
            outcome_index=batch["outcome_index"],
            context_int_mask=batch["context_int_mask"],
        )
        # Every group has Nq_g rows, so the mean over the B * G elements is
        # the per-task average.
        loss = self.model.calculate_loss(
            model_output=model_output,
            target=batch["target"],
            outcome_index=batch["outcome_index"],
            test=test,
        )
        return {"loss": loss, "model_output": model_output}

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Train step that skips non-finite steps and saves the first few spike batches."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._forward_pass(batch)["loss"]
        if not th.isfinite(loss):
            return {"loss": float("nan"), "grad_norm": float("nan"), "skipped": 1.0}
        loss.backward()
        clip = self.trainingconfig.gradient_clip_val
        grad_norm = float(th.nn.utils.clip_grad_norm_(
            self.model.parameters(), clip if clip is not None else float("inf")))
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": loss.item(), "grad_norm": grad_norm, "skipped": 1.0}
        if grad_norm > 1000 and getattr(self, "n_spikes_saved", 0) < 5:
            self.n_spikes_saved = getattr(self, "n_spikes_saved", 0) + 1
            path = Path(self.loggingconfig.save_dir) / f"spike_step_{self.global_step}.pt"
            th.save({"batch": {k: v.detach().cpu() if th.is_tensor(v) else v
                               for k, v in batch.items()},
                     "loss": loss.item(), "grad_norm": grad_norm,
                     "step": self.global_step}, path)
            print(f"[spike] step {self.global_step} grad_norm {grad_norm:.0f}", flush=True)
        self.optimizer.step()
        return {"loss": loss.item(), "grad_norm": grad_norm, "skipped": 0.0}

    def validate(self) -> Dict[str, float]:
        self._eval_batches = True
        try:
            return super().validate()
        finally:
            self._eval_batches = False

    def test_epoch(self, test_loader, metric_dict):
        """Test loss overall and by stratum: causal direction (T -> Y),
        group type (seen / unseen), has_obs, K (0, 1, 2+) and G."""
        self._eval_batches = True
        losses = defaultdict(list)
        try:
            with th.no_grad():
                self.model.eval()
                self.model.to(self.eval_dtype)
                self.model.to(self.device)
                for batch in tqdm(test_loader, desc="Test", leave=False):
                    prep = self._prepare_batch(batch, dtype=self.eval_dtype)
                    directions = get_causal_direction(
                        graphs=prep["causal_graphs"],
                        intvn_indices=prep["intervention_index"],
                        outcome_indices=prep["outcome_index"],
                    )
                    # Per-element losses [B * G].
                    loss = self._forward_pass(prep, test=True)["loss"]
                    G = prep["G"]
                    for i, val in enumerate(loss.tolist()):
                        m = prep["meta"][i // G]
                        k = m["K"]
                        for key in (
                            "all",
                            directions[i],
                            f"group_type={prep['group_types'][i]}",
                            f"has_obs={m['has_obs']}",
                            f"K={'2+' if k >= 2 else k}",
                            f"G={G}",
                        ):
                            losses[key].append(val)
        finally:
            self._eval_batches = False

        losses_np = {k: np.array(v) for k, v in losses.items()}
        # Upstream's keys first (None when a stratum is empty), then the rest.
        for key in ["all", "downstream", "upstream", "independent"]:
            vals = losses_np.pop(key, None)
            metric_dict[f"test_loss_{key}_mean"] = (
                None if vals is None or len(vals) == 0 else float(vals.mean())
            )
            metric_dict[f"test_loss_{key}_std"] = (
                None if vals is None or len(vals) == 0 else float(vals.std())
            )
        for key in sorted(losses_np):
            metric_dict[f"test_loss_{key}_mean"] = float(losses_np[key].mean())
            metric_dict[f"test_loss_{key}_std"] = float(losses_np[key].std())
            metric_dict[f"test_count_{key}"] = int(len(losses_np[key]))
        return metric_dict
