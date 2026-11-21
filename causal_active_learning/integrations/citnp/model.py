"""`CausalPredictiveModel` for CITNP joint models, through
`CITNP/evaluation/model_api.py` only.

- `predict_query`: the marginal of the outcome in the model's joint prediction
  (`predict_mixture_batched`'s joint dispatch), so fantasies and the query
  come from one set of beliefs.
- `sample_fantasies`: `predict_joint_mixture_batched` and `sample_joint_rows`
  with explicit noise and per-query do-values, row by row (autoregressive).
- Scale statistics are recomputed from every (fantasy) context inside
  `model_api` (`_batched_model_inputs` -> `_normalisation_stats`), exactly as
  in training.
"""

from collections.abc import Sequence

import torch

from causal_active_learning.context import stack_contexts
from causal_active_learning.distributions import UnivariateGaussianMixture
from CITNP.evaluation.model_api import (
    JointPrediction,
    JointSampleNoise,
    draw_joint_sample_noise,
    joint_covariance_rank,
    predict_joint_mixture_batched,
    predict_mixture_batched,
    sample_joint_rows,
)

FANTASY_ROW_SAMPLING = ("autoregressive", "independent")


def _cpu_joint(jp: JointPrediction) -> JointPrediction:
    return JointPrediction(
        component_mean=jp.component_mean.cpu(),
        covariance_factor=jp.covariance_factor.cpu(),
        covariance_diagonal=jp.covariance_diagonal.cpu(),
        log_weights=jp.log_weights.cpu(),
        intervened_mask=jp.intervened_mask.cpu(),
    )


def _concatenate(parts: list[JointPrediction]) -> JointPrediction:
    return JointPrediction(
        component_mean=torch.cat([p.component_mean for p in parts]),
        covariance_factor=torch.cat([p.covariance_factor for p in parts]),
        covariance_diagonal=torch.cat([p.covariance_diagonal for p in parts]),
        log_weights=torch.cat([p.log_weights for p in parts]),
        intervened_mask=torch.cat([p.intervened_mask for p in parts]),
    )


def _per_query_noise(noise: Sequence[JointSampleNoise], row: int, lead):
    """Row `row` of each noise [S, R, ...], stacked and reshaped so that
    `sample_joint_rows` reads noise b for query b."""
    return JointSampleNoise(
        component_uniform=torch.stack(
            [n.component_uniform[:, row] for n in noise]
        ).reshape(*lead),
        factor_normal=torch.stack(
            [n.factor_normal[:, row] for n in noise]
        ).reshape(*lead, -1),
        residual_normal=torch.stack(
            [n.residual_normal[:, row] for n in noise]
        ).reshape(*lead, -1),
    )


class CITNPPredictiveModel:
    """`CausalPredictiveModel` backed by `CITNP.evaluation.model_api` (a joint
    model, prediction_head: joint).

    Args:
    ----------
    model : torch.nn.Module
        A loaded joint model (`model_api.load_model`).

    device : str | torch.device
        Where the model runs; outputs are returned on the CPU.

    fantasy_row_sampling : str
        "autoregressive" (default; the exact predictive chain rule) or
        "independent" (all rows from one prediction; for comparison only, it
        overstates how much a multi-row acquisition reveals).

    inference_batch_size : int
        Contexts per forward pass.

    minimum_observational_rows : int
        Fewest observational rows a context may have (default 10, the check
        of CITNP's training-time standardisation, `_normalisation_stats`).
        Lower it only knowingly: the per-variable standardisation is then
        estimated from fewer rows than training ever used.

    supports_observational_prediction : bool
        True only for a model trained with observational queries (plan Part
        B); every current CITNP checkpoint has False.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device,
        *,
        fantasy_row_sampling: str = "autoregressive",
        inference_batch_size: int = 128,
        minimum_observational_rows: int = 10,
        supports_observational_prediction: bool = False,
    ):
        if getattr(model, "prediction_head", "marginal") != "joint":
            raise ValueError(
                "active learning needs a joint model (prediction_head: joint): "
                "the fantasies come from its joint prediction"
            )
        if fantasy_row_sampling not in FANTASY_ROW_SAMPLING:
            raise ValueError(
                f"fantasy_row_sampling must be one of {FANTASY_ROW_SAMPLING}"
            )
        self.model = model.eval()
        self.device = device
        self.covariance_rank = joint_covariance_rank(model)
        self.fantasy_row_sampling = fantasy_row_sampling
        self.inference_batch_size = int(inference_batch_size)
        self.minimum_observational_rows = int(minimum_observational_rows)
        self.supports_observational_prediction = bool(
            supports_observational_prediction
        )

    # ------------------------------------------------------------------ query
    def predict_query(self, contexts, treatment, outcome, treatment_values):
        psi = torch.as_tensor(treatment_values, dtype=torch.float64)
        parts = []
        for start in range(0, len(contexts), self.inference_batch_size):
            values, flags = stack_contexts(
                contexts[start : start + self.inference_batch_size]
            )
            mean, sd, log_weights = predict_mixture_batched(
                self.model,
                values,
                flags,
                int(treatment),
                int(outcome),
                psi,
                device=self.device,
                min_rows=self.minimum_observational_rows,
            )
            parts.append(
                UnivariateGaussianMixture(mean.cpu(), sd.cpu(), log_weights.cpu())
            )
        return UnivariateGaussianMixture.concatenate(parts)

    # -------------------------------------------------------------- fantasies
    def draw_fantasy_noise(
        self, num_variables, num_fantasies, num_rows, generator
    ):
        return draw_joint_sample_noise(
            num_fantasies,
            self.covariance_rank,
            num_variables,
            generator,
            num_rows=num_rows,
        )

    def _joint(self, contexts, nodes, values) -> JointPrediction:
        """One joint prediction per context (Q = B, n_t = 1)."""
        parts = []
        for start in range(0, len(contexts), self.inference_batch_size):
            stop = start + self.inference_batch_size
            ctx_values, flags = stack_contexts(contexts[start:stop])
            parts.append(
                _cpu_joint(
                    predict_joint_mixture_batched(
                        self.model,
                        ctx_values,
                        flags,
                        nodes[start:stop],
                        values[start:stop, None],
                        device=self.device,
                        min_rows=self.minimum_observational_rows,
                    )
                )
            )
        return _concatenate(parts)

    def sample_fantasies(self, contexts, acquisitions, noise):
        acquisitions = list(acquisitions)
        if any(a.is_observational for a in acquisitions):
            raise NotImplementedError(
                "model_api offers no observational prediction (the joint head "
                "always predicts p(X_-T | do(X_T = t)), with one intervened "
                "variable per query)"
            )
        R = acquisitions[0].num_rows
        if any(a.num_rows != R for a in acquisitions):
            raise ValueError("all acquisitions must have the same number of rows")
        C = len(acquisitions)
        S = int(noise[0].component_uniform.shape[0])
        nodes = torch.tensor([int(a.target_node or 0) for a in acquisitions])
        values = torch.tensor(
            [float(a.intervention_value or 0.0) for a in acquisitions],
            dtype=torch.float64,
        )
        joint = self._joint(list(contexts), nodes, values)
        if self.fantasy_row_sampling == "independent" and R > 1:
            flat = [
                JointSampleNoise(
                    n.component_uniform[:, :R].T.reshape(-1),
                    n.factor_normal[:, :R].transpose(0, 1).reshape(R * S, -1),
                    n.residual_normal[:, :R].transpose(0, 1).reshape(R * S, -1),
                )
                for n in noise
            ]  # row-major: sample index r * S + s
            stacked = JointSampleNoise(
                torch.stack([n.component_uniform for n in flat])[:, None],
                torch.stack([n.factor_normal for n in flat])[:, None],
                torch.stack([n.residual_normal for n in flat])[:, None],
            )
            drawn = sample_joint_rows(
                joint, values[:, None], R * S, noise=stacked
            )
            return drawn[:, 0].reshape(C, R, S, -1).transpose(1, 2).contiguous()
        first = sample_joint_rows(
            joint, values[:, None], S, noise=_per_query_noise(noise, 0, (C, 1, S))
        )[:, 0]  # [C, S, D]
        rows = [first]
        for r in range(1, R):
            so_far = torch.stack(rows, dim=2)  # [C, S, r, D]
            fantasy = [
                contexts[c].append(so_far[c, s], int(nodes[c]))
                for c in range(C)
                for s in range(S)
            ]
            joint_r = self._joint(
                fantasy, nodes.repeat_interleave(S), values.repeat_interleave(S)
            )  # one prediction per fantasy context: Q = C * S
            drawn = sample_joint_rows(
                joint_r,
                values.repeat_interleave(S)[:, None],
                1,
                noise=_per_query_noise(noise, r, (C * S, 1, 1)),
            )
            rows.append(drawn.reshape(C, S, -1))
        return torch.stack(rows, dim=2)  # [C, S, R, D]
