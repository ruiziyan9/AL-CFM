"""Contexts, context augmentation and fantasy rows.

A fantasy is a hypothetical outcome of a candidate experiment, drawn from the
model's own joint prediction p(X_{-i} | do(X_i = v), D), with column i set to
v. Appending it to the real context D gives a fantasy context D^(s); the
query predicted on D^(s) shows how the experiment could change the model's
answer (section 3 of the plan). Fantasy contexts are never kept.

Fantasy noise is passed in explicitly (`JointSampleNoise`), so the same
standard noise can be reused for every value of one node (common random
numbers: the fantasy rows then move smoothly with the value).
"""

from dataclasses import dataclass

import torch

from CITNP.active_learning.config import CandidateExperiment
from CITNP.active_learning.predictive_model import (
    PredictiveModel,
    predict_experiment_joint_in_chunks,
)
from CITNP.evaluation.model_api import JointSampleNoise, sample_joint_rows


@dataclass
class ExperimentContext:
    """One real context on the raw data scale.

    Attributes:
    -----------
    - values: [N, D] float64.
    - intervened: [N, D] bool, True where the cell was set by do().
    """

    values: torch.Tensor
    intervened: torch.Tensor

    def __post_init__(self):
        self.values = torch.as_tensor(self.values, dtype=torch.float64).cpu()
        self.intervened = torch.as_tensor(self.intervened, dtype=torch.bool)
        self.intervened = self.intervened.cpu()
        if self.values.ndim != 2 or self.values.shape != self.intervened.shape:
            raise ValueError("values and intervened must both be [N, D]")

    @property
    def num_rows(self) -> int:
        """N."""
        return int(self.values.shape[0])

    @property
    def num_variables(self) -> int:
        """D."""
        return int(self.values.shape[1])

    def append(
        self, new_rows: torch.Tensor, target_node: int | None
    ) -> "ExperimentContext":
        """A new context with `new_rows` [R, D] appended (see
        `append_rows_to_context`); this one is unchanged."""
        values, intervened = append_rows_to_context(
            self.values, self.intervened, new_rows, target_node
        )
        return ExperimentContext(values, intervened)

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`size` copies as ([size, N, D] values, [size, N, D] flags)."""
        return (
            self.values[None].expand(size, -1, -1),
            self.intervened[None].expand(size, -1, -1),
        )


def append_rows_to_context(
    context_values: torch.Tensor,
    context_intervened: torch.Tensor,
    new_rows: torch.Tensor,
    target_node: int | None | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append rows to contexts; the new rows are flagged as intervened on
    `target_node` and on nothing else (on nothing at all when None or < 0:
    an observational row). Rows go at the end: context rows are
    exchangeable for the model.

    The model's other context inputs (standardisation and scale features)
    are derived from values and flags inside `model_api` (`_model_inputs`,
    `_batched_model_inputs`, through the training code's
    `_normalisation_stats`), so they are recomputed from the augmented
    context by the existing code path.

    Args:
    ----------
    context_values : torch.Tensor
        [..., N, D] float.

    context_intervened : torch.Tensor
        [..., N, D] bool.

    new_rows : torch.Tensor
        [..., R, D] float, leading axes as the context's.

    target_node : int | None | torch.Tensor
        One node for all contexts, or a long tensor [...] with one node per
        context (-1 = observational).

    Returns:
    ----------
    (values [..., N + R, D], intervened [..., N + R, D]).
    """
    new_rows = torch.as_tensor(new_rows, dtype=context_values.dtype)
    D = context_values.shape[-1]
    flags = torch.zeros(new_rows.shape, dtype=torch.bool)
    if isinstance(target_node, torch.Tensor):
        node = target_node.to(torch.long)
        one_hot = torch.nn.functional.one_hot(node.clamp(min=0), D).bool()
        one_hot = one_hot & (node >= 0)[..., None]
        flags = flags | one_hot[..., None, :]
    elif target_node is not None:
        flags[..., int(target_node)] = True
    values = torch.cat([context_values, new_rows], dim=-2)
    intervened = torch.cat([context_intervened, flags], dim=-2)
    return values, intervened


def _node_code(candidate: CandidateExperiment) -> int:
    """Target node, or -1 for an observational experiment."""
    return -1 if candidate.target_node is None else candidate.target_node


def _noise_rows(noise: JointSampleNoise, num_rows: int) -> list[JointSampleNoise]:
    """Per-row noise [S, ...] of a noise with or without a row axis."""
    if noise.component_uniform.ndim == 1:
        if num_rows != 1:
            raise ValueError(
                "a multi-row experiment needs noise with a row axis "
                "(draw_joint_sample_noise(num_rows=...))"
            )
        return [noise]
    if noise.component_uniform.shape[1] < num_rows:
        raise ValueError("the noise has fewer rows than the experiment")
    return [noise.row(r) for r in range(num_rows)]


def _stack_noise(noises: list[JointSampleNoise]) -> JointSampleNoise:
    """Stack per-candidate noise [S, ...] to [C, 1, S, ...], one noise per
    query of a `JointPrediction` with Q = C and n_t = 1."""
    return JointSampleNoise(
        component_uniform=torch.stack([n.component_uniform for n in noises])[
            :, None
        ],
        factor_normal=torch.stack([n.factor_normal for n in noises])[:, None],
        residual_normal=torch.stack([n.residual_normal for n in noises])[:, None],
    )


def sample_fantasy_rows(
    predictive_model: PredictiveModel,
    context: ExperimentContext,
    candidates: list[CandidateExperiment],
    noises: list[JointSampleNoise],
    mode: str = "autoregressive",
    batch_size: int = 128,
) -> torch.Tensor:
    """Fantasy outcomes of each candidate experiment on `context`.

    Row r of fantasy s of candidate c uses row r of noise s of `noises[c]`;
    the interventional column equals the do-value exactly.

    - One row (R = 1): one joint prediction per candidate on the real
      context, then S rows from it (`sample_joint_rows(noise=...)`).
    - R > 1, mode "autoregressive" (default): draw row 1, append it, predict
      again on every fantasy context, draw row 2, and so on: the exact
      predictive chain rule, batched over candidates and fantasies (R model
      calls per batch).
    - R > 1, mode "independent": R rows i.i.d. from one prediction. Cheap,
      but it ignores that the rows of one experiment share the unknown
      mechanism, so it misstates the information in a multi-row experiment.

    Args:
    ----------
    predictive_model : PredictiveModel
        Source of the joint predictions.

    context : ExperimentContext
        The real context [N, D].

    candidates : list[CandidateExperiment]
        C candidates, all with the same rows_per_experiment R (an
        observational candidate needs `predict_observational_joint`).

    noises : list[JointSampleNoise]
        One per candidate, S samples each ([S, ...] fields for R = 1, or
        [S, R', ...] with R' >= R).

    mode : str
        "autoregressive" or "independent".

    batch_size : int
        Contexts per forward pass.

    Returns:
    ----------
    torch.Tensor
        [C, S, R, D] float64 rows on the raw scale.
    """
    if mode not in ("autoregressive", "independent"):
        raise ValueError(f"unknown fantasy mode {mode!r}")
    if len(candidates) != len(noises) or not candidates:
        raise ValueError("give one noise per candidate (and >= 1 candidate)")
    R = candidates[0].rows_per_experiment
    if any(c.rows_per_experiment != R for c in candidates):
        raise ValueError("all candidates must have the same rows_per_experiment")
    observational = [c.is_observational for c in candidates]
    if any(observational) and not all(observational):
        # one call per kind, reassembled in the given order
        out = [None] * len(candidates)
        for kind in (True, False):
            idx = [i for i, o in enumerate(observational) if o == kind]
            rows = sample_fantasy_rows(
                predictive_model, context, [candidates[i] for i in idx],
                [noises[i] for i in idx], mode, batch_size,
            )  # fmt: skip
            for j, i in enumerate(idx):
                out[i] = rows[j]
        return torch.stack(out)  # type: ignore[arg-type]

    C = len(candidates)
    S = noises[0].num_samples
    if any(n.num_samples != S for n in noises):
        raise ValueError("all noises must have the same number of samples")
    nodes = torch.tensor([_node_code(c) for c in candidates], dtype=torch.long)
    values = torch.tensor(
        [0.0 if c.intervention_value is None else c.intervention_value
         for c in candidates],
        dtype=torch.float64,
    )  # fmt: skip
    per_row = [_noise_rows(n, R) for n in noises]  # [C][R] -> [S, ...]

    base_values, base_flags = context.batch(C)
    joint = predict_experiment_joint_in_chunks(
        predictive_model, base_values, base_flags, nodes, values, batch_size
    )  # Q = C, n_t = 1
    if mode == "independent" and R > 1:
        flat = [
            JointSampleNoise(
                torch.cat([rn.component_uniform for rn in rows]),
                torch.cat([rn.factor_normal for rn in rows]),
                torch.cat([rn.residual_normal for rn in rows]),
            )
            for rows in per_row
        ]  # row-major: sample index r * S + s
        drawn = sample_joint_rows(
            joint, values[:, None], R * S, noise=_stack_noise(flat)
        )[:, 0]  # [C, R * S, D]
        return drawn.reshape(C, R, S, -1).transpose(1, 2).contiguous()

    first = sample_joint_rows(
        joint, values[:, None], S, noise=_stack_noise([r[0] for r in per_row])
    )[:, 0]  # [C, S, D]
    rows = [first]
    D = context.num_variables
    for r in range(1, R):
        fantasy_values, fantasy_flags = context.batch(C * S)
        so_far = torch.stack(rows, dim=2).reshape(C * S, r, D)
        fantasy_values, fantasy_flags = append_rows_to_context(
            fantasy_values, fantasy_flags, so_far,
            nodes.repeat_interleave(S),
        )  # fmt: skip
        # one prediction per fantasy context: Q = C * S
        joint_r = predict_experiment_joint_in_chunks(
            predictive_model, fantasy_values, fantasy_flags,
            nodes.repeat_interleave(S), values.repeat_interleave(S),
            batch_size,
        )  # fmt: skip
        noise_r = JointSampleNoise(
            component_uniform=torch.stack(
                [rows_c[r].component_uniform for rows_c in per_row]
            ).reshape(C * S, 1, 1),
            factor_normal=torch.stack(
                [rows_c[r].factor_normal for rows_c in per_row]
            ).reshape(C * S, 1, 1, -1),
            residual_normal=torch.stack(
                [rows_c[r].residual_normal for rows_c in per_row]
            ).reshape(C * S, 1, 1, -1),
        )
        drawn = sample_joint_rows(
            joint_r, values.repeat_interleave(S)[:, None], 1, noise=noise_r
        )  # [C * S, 1, 1, D]
        rows.append(drawn.reshape(C, S, D))
    return torch.stack(rows, dim=2)  # [C, S, R, D]
