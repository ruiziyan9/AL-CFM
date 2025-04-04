from abc import ABC, abstractmethod
from typing import Optional

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
