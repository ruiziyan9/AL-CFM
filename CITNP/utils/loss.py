from abc import ABC, abstractmethod
from typing import Optional

import einops
import torch

from CITNP.utils.outputs import ModelOutput


class BaseLoss(ABC):
    def __init__(self, mean_loss_across_samples: bool, reduce: str):
        self.mean_loss_across_samples = mean_loss_across_samples
        assert reduce in ["mean", "sum"]
        self.reduce = reduce

    @abstractmethod
    def calculate_loss(
        self, model_output: ModelOutput, trgt_outcome: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class CNPLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_loss(
        self, model_output: ModelOutput, trgt_outcome: torch.Tensor
    ) -> torch.Tensor:
        pred_mean = model_output.pred_mean
        pred_std = model_output.pred_std

        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)
        log_prob = distribution.log_prob(trgt_outcome)

        if self.mean_loss_across_samples:
            log_prob_all_samples = log_prob.mean(1).sum(-1)
        else:
            log_prob_all_samples = log_prob.sum(1).sum(-1)

        if self.reduce == "mean":
            log_prob_allbatch = log_prob_all_samples.mean(0)
        elif self.reduce == "sum":
            log_prob_allbatch = log_prob_all_samples.sum(0)

        neg_log_prob_allbatch = -log_prob_allbatch
        return neg_log_prob_allbatch


class MixtureGaussianLoss(BaseLoss):
    def __init__(
        self,
        num_mixture_components: Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mixture_components = num_mixture_components

    def calculate_loss(
        self, model_output: ModelOutput, trgt_outcome: torch.Tensor
    ) -> torch.Tensor:
        pred_mean = model_output.pred_mean
        pred_std = model_output.pred_std
        weights = model_output.weights
        _, _, num_outcome_nodes, num_components = pred_mean.shape
        assert num_outcome_nodes == trgt_outcome.shape[-1], (
            "Mismatch between number of outcome nodes in model output and target outcome"
        )

        # Expand target to match mixture dimensions
        target_expanded = einops.repeat(
            trgt_outcome, "b nt d -> b nt d K", K=num_components
        )

        # Define the Gaussian distributions for each component
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)

        # Compute log probabilities for all components
        log_prob = distribution.log_prob(
            target_expanded
        )  # Shape: (batch_size, num_samples, K)

        assert weights is not None, "weights must not be None"
        # Weighted log probabilities with mixture weights
        weighted_log_prob = torch.logsumexp(
            log_prob + torch.log(weights + 1e-12), dim=-1
        )  # Shape: (batch_size, num_samples)

        # Reduce over the samples
        if self.mean_loss_across_samples:
            log_prob_all_samples = weighted_log_prob.mean(1).sum(-1)
        else:
            log_prob_all_samples = weighted_log_prob.sum(1).sum(-1)

        if self.reduce == "mean":
            log_prob_allbatch = log_prob_all_samples.mean(0)
        elif self.reduce == "sum":
            log_prob_allbatch = log_prob_all_samples.sum(0)

        neg_log_prob_allbatch = -log_prob_allbatch
        return neg_log_prob_allbatch


class NLLLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_loss(
        self, model_output: ModelOutput, trgt_outcome: torch.Tensor
    ) -> torch.Tensor:
        pred_mean = model_output.pred_mean
        pred_std = model_output.pred_std

        num_z_samples = pred_mean.size(0)
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)

        likelihood = distribution.log_prob(trgt_outcome)
        logL = torch.log(
            torch.tensor(
                num_z_samples,
                device=likelihood.device,
                dtype=likelihood.dtype,
            )
        )
        likelihood_z_meaned = torch.logsumexp(likelihood, dim=0) - logL

        if self.mean_loss_across_samples:
            likelihood_allsample = likelihood_z_meaned.mean(1).sum(-1)
        else:
            likelihood_allsample = likelihood_z_meaned.sum(1).sum(-1)

        if self.reduce == "mean":
            likelihood_allbatch = likelihood_allsample.mean(0)
        elif self.reduce == "sum":
            likelihood_allbatch = likelihood_allsample.sum(0)

        neglikelihood_allbatch = -likelihood_allbatch
        return neglikelihood_allbatch
