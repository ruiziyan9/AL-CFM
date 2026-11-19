"""
Classes containing structured outputs for the model classes.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelOutput:
    pred_mean: torch.Tensor
    pred_std: torch.Tensor
    weights: Optional[torch.Tensor] = None


@dataclass
class JointModelOutput:
    """Output of the joint prediction head (`JointMoGPredictor`): per query
    row, a K-component mixture of D-variate Gaussians over all variables,

        p(x) = sum_k softmax(mixture_logits)_k N(x; component_mean_k, Sigma_k),
        Sigma_k = covariance_factor_k covariance_factor_k^T
                  + diag(covariance_diagonal_k),

    on the standardised scale. B is the flattened (task, group) batch, Nq the
    query rows, K the mixture components, D the variables, R the covariance
    rank.

    The head outputs parameters for every variable, including the intervened
    ones; those are auxiliary: the loss never uses them and they are not a
    prediction for that variable (it is fixed by do()). The distribution of
    interest is the block of the non-intervened variables.

    Attributes:
    -----------
    - component_mean: [B, Nq, K, D].
    - covariance_factor: [B, Nq, K, D, R].
    - covariance_diagonal: [B, Nq, K, D], strictly positive.
    - mixture_logits: [B, Nq, K], unnormalised log mixture weights.
    - intervened_mask: [B, D] bool, True = set by do().
    """

    component_mean: torch.Tensor
    covariance_factor: torch.Tensor
    covariance_diagonal: torch.Tensor
    mixture_logits: torch.Tensor
    intervened_mask: torch.Tensor
