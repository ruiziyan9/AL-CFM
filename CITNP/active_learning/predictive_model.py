"""The interface between the active learning loop and a predictive model.

The loop needs two kinds of prediction, both on the RAW data scale in
float64, for a batch of contexts of equal length:

- the query, p(X_Y | do(X_T = psi), context) on a grid of psi values
  (`predict_query`, a univariate Gaussian mixture per context and psi);
- the outcome of an experiment, p(X_{-i} | do(X_i = v), context)
  (`predict_intervention_joint`, a `JointPrediction` with one query per
  context), from which fantasy rows are drawn.

`ModelApiPredictiveModel` implements this with `CITNP.evaluation.model_api`
(any model that `model_api` supports with a joint head); the tests implement
it with closed-form stubs. Both predictions come from the same model, so the
fantasies and the query express the same beliefs.
"""

from dataclasses import dataclass
from typing import Protocol

import torch

from CITNP.evaluation.model_api import (
    JointPrediction,
    joint_covariance_rank,
    predict_joint_mixture_batched,
    predict_mixture_batched,
)


@dataclass
class QueryMixture:
    """Univariate Gaussian mixtures of the query variable on the raw scale,
    float64:
        p(y) = sum_k exp(log_weights_k) N(y; mean_k, standard_deviation_k^2).

    Attributes:
    -----------
    - mean: [..., V, K], V the number of query treatment values.
    - standard_deviation: [..., V, K].
    - log_weights: [..., V, K], normalised over K.
    """

    mean: torch.Tensor
    standard_deviation: torch.Tensor
    log_weights: torch.Tensor

    def __getitem__(self, index) -> "QueryMixture":
        """Index the leading axes of all three tensors."""
        return QueryMixture(
            self.mean[index], self.standard_deviation[index],
            self.log_weights[index],
        )  # fmt: skip

    def reshape(self, *leading: int) -> "QueryMixture":
        """Reshape the axes before (V, K) to `leading`."""
        tail = self.mean.shape[-2:]
        return QueryMixture(
            self.mean.reshape(*leading, *tail),
            self.standard_deviation.reshape(*leading, *tail),
            self.log_weights.reshape(*leading, *tail),
        )

    @staticmethod
    def concatenate(parts: list["QueryMixture"]) -> "QueryMixture":
        """Concatenate along the first axis."""
        return QueryMixture(
            torch.cat([p.mean for p in parts]),
            torch.cat([p.standard_deviation for p in parts]),
            torch.cat([p.log_weights for p in parts]),
        )

    @staticmethod
    def gaussian(mean: torch.Tensor, standard_deviation: torch.Tensor):
        """One-component mixtures from means and standard deviations [..., V]."""
        mean = torch.as_tensor(mean, dtype=torch.float64)
        sd = torch.as_tensor(standard_deviation, dtype=torch.float64)
        return QueryMixture(
            mean[..., None], sd[..., None], torch.zeros_like(mean)[..., None]
        )


class PredictiveModel(Protocol):
    """What the active learning loop needs from a model.

    Contexts are float64 values [B, N, D] and bool intervened flags [B, N, D]
    on the CPU (row n of context b intervened on node j iff flag [b, n, j]).
    """

    num_variables: int
    covariance_rank: int

    def predict_query(
        self,
        context_values: torch.Tensor,
        context_intervened: torch.Tensor,
        treatment: int,
        outcome: int,
        treatment_values: torch.Tensor,
    ) -> QueryMixture:
        """p(X_outcome | do(X_treatment = psi), context_b) for every context b
        and psi in treatment_values [V]: a `QueryMixture` [B, V, K]."""
        ...

    def predict_intervention_joint(
        self,
        context_values: torch.Tensor,
        context_intervened: torch.Tensor,
        target_nodes: torch.Tensor,
        intervention_values: torch.Tensor,
    ) -> JointPrediction:
        """p(X_{-i_b} | do(X_{i_b} = v_b), context_b) for every context b, with
        target_nodes [B] long and intervention_values [B] float64: a
        `JointPrediction` with Q = B queries and n_t = 1 (CPU, float64)."""
        ...

    def predict_observational_joint(
        self, context_values: torch.Tensor, context_intervened: torch.Tensor
    ) -> JointPrediction:
        """p(X | context_b), the joint of a new observational row (Q = B,
        n_t = 1, no intervened variable). May raise NotImplementedError."""
        ...


def _cpu_joint(jp: JointPrediction) -> JointPrediction:
    return JointPrediction(
        component_mean=jp.component_mean.cpu(),
        covariance_factor=jp.covariance_factor.cpu(),
        covariance_diagonal=jp.covariance_diagonal.cpu(),
        log_weights=jp.log_weights.cpu(),
        intervened_mask=jp.intervened_mask.cpu(),
    )


class ModelApiPredictiveModel:
    """`PredictiveModel` backed by `CITNP.evaluation.model_api` (a joint
    model, prediction_head: joint). The query is the marginal of the outcome
    in the same model's joint prediction (`predict_mixture_batched`'s joint
    dispatch), so query and fantasies come from one set of beliefs.

    Args:
    ----------
    model : torch.nn.Module
        A loaded joint model (`model_api.load_model`), in eval mode.

    num_variables : int
        D.

    device : str | torch.device
        Where the model runs; outputs are returned on the CPU.
    """

    def __init__(self, model: torch.nn.Module, num_variables: int, device):
        if getattr(model, "prediction_head", "marginal") != "joint":
            raise ValueError(
                "active learning needs a joint model (prediction_head: joint): "
                "the fantasy rows come from its joint prediction"
            )
        self.model = model.eval()
        self.num_variables = int(num_variables)
        self.covariance_rank = joint_covariance_rank(model)
        self.device = device

    def predict_query(
        self, context_values, context_intervened, treatment, outcome,
        treatment_values,
    ) -> QueryMixture:  # fmt: skip
        mean, sd, log_weights = predict_mixture_batched(
            self.model, context_values, context_intervened, int(treatment),
            int(outcome), torch.as_tensor(treatment_values, dtype=torch.float64),
            device=self.device,
        )  # fmt: skip
        return QueryMixture(mean.cpu(), sd.cpu(), log_weights.cpu())

    def predict_intervention_joint(
        self, context_values, context_intervened, target_nodes,
        intervention_values,
    ) -> JointPrediction:  # fmt: skip
        values = torch.as_tensor(intervention_values, dtype=torch.float64)
        joint = predict_joint_mixture_batched(
            self.model, context_values, context_intervened,
            torch.as_tensor(target_nodes, dtype=torch.long), values[:, None],
            device=self.device,
        )  # fmt: skip
        return _cpu_joint(joint)

    def predict_observational_joint(self, context_values, context_intervened):
        raise NotImplementedError(
            "model_api offers no observational prediction (the joint head "
            "always predicts p(X_-T | do(X_T = t)), with one intervened "
            "variable per query), so the observational candidate is not "
            "available for this model"
        )


# --------------------------------------------------------------------------- #
# Chunked calls (inference_batch_size contexts per forward pass)
# --------------------------------------------------------------------------- #
def predict_query_in_chunks(
    predictive_model: PredictiveModel,
    context_values: torch.Tensor,
    context_intervened: torch.Tensor,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    batch_size: int,
) -> QueryMixture:
    """`predict_query` on [B, N, D] contexts in chunks of `batch_size`
    under torch.no_grad(); returns a `QueryMixture` [B, V, K]."""
    parts = []
    with torch.no_grad():
        for start in range(0, context_values.shape[0], batch_size):
            stop = start + batch_size
            parts.append(
                predictive_model.predict_query(
                    context_values[start:stop],
                    context_intervened[start:stop],
                    treatment,
                    outcome,
                    treatment_values,
                )  # fmt: skip
            )
    return QueryMixture.concatenate(parts)


def concatenate_joint_predictions(
    parts: list[JointPrediction],
) -> JointPrediction:
    """Concatenate `JointPrediction`s along the query axis."""
    return JointPrediction(
        component_mean=torch.cat([p.component_mean for p in parts]),
        covariance_factor=torch.cat([p.covariance_factor for p in parts]),
        covariance_diagonal=torch.cat([p.covariance_diagonal for p in parts]),
        log_weights=torch.cat([p.log_weights for p in parts]),
        intervened_mask=torch.cat([p.intervened_mask for p in parts]),
    )


def predict_experiment_joint_in_chunks(
    predictive_model: PredictiveModel,
    context_values: torch.Tensor,
    context_intervened: torch.Tensor,
    target_nodes: torch.Tensor,
    intervention_values: torch.Tensor,
    batch_size: int,
) -> JointPrediction:
    """The joint of one new row per context b: interventional
    (`predict_intervention_joint`) where target_nodes[b] >= 0, observational
    (`predict_observational_joint`) where target_nodes[b] < 0, in chunks of
    `batch_size`. target_nodes [B] long, intervention_values [B] float64
    (ignored for observational rows). Returns Q = B, n_t = 1."""
    target_nodes = torch.as_tensor(target_nodes, dtype=torch.long)
    observational = target_nodes < 0
    if observational.any() and not observational.all():
        raise ValueError("a chunked call is all observational or all not")
    parts = []
    with torch.no_grad():
        for start in range(0, context_values.shape[0], batch_size):
            stop = start + batch_size
            if observational.all():
                parts.append(
                    predictive_model.predict_observational_joint(
                        context_values[start:stop],
                        context_intervened[start:stop],
                    )
                )
            else:
                parts.append(
                    predictive_model.predict_intervention_joint(
                        context_values[start:stop],
                        context_intervened[start:stop],
                        target_nodes[start:stop],
                        intervention_values[start:stop],
                    )
                )
    return concatenate_joint_predictions(parts)
