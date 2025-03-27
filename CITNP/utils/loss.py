from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseLoss(ABC):

    @abstractmethod
    def calculate_loss(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target: torch.Tensor,
        z_mean: Optional[torch.Tensor] = None,
        z_std: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError


class NLLLoss(BaseLoss):

    def __init__(self, mean_loss_across_samples: bool):
        self.mean_loss_across_samples = mean_loss_across_samples
        pass

    def calculate_loss(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target: torch.Tensor,
        z_mean: Optional[torch.Tensor] = None,
        z_std: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        num_z_samples = pred_mean.size(0)
        # During evaluation we only care about the predictive likelihood
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)

        neglikelihood = -distribution.log_prob(target)
        logL = torch.log(
            torch.tensor(
                num_z_samples,
                device=neglikelihood.device,
                dtype=neglikelihood.dtype,
            )
        )
        neglikelihood_z_meaned = torch.logsumexp(neglikelihood, dim=0) - logL
        if self.mean_loss_across_samples:
            neglikelihood_allsample = neglikelihood_z_meaned.mean(1).sum(-1)
        else:
            neglikelihood_allsample = neglikelihood_z_meaned.sum(1).sum(-1)
        neglikelihood_fullmeaned = neglikelihood_allsample.mean(0).mean()
        return neglikelihood_fullmeaned


class VILoss(BaseLoss):

    def __init__(self, mean_loss_across_samples: bool):
        self.mean_loss_across_samples = mean_loss_across_samples
        pass

    def calculate_loss(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target: torch.Tensor,
        z_mean: torch.Tensor,
        z_std: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # Data term
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)
        neglikelihood = -distribution.log_prob(target)
        neglikelihood_z_meaned = neglikelihood.mean(0)
        if self.mean_loss_across_samples:
            neglikelihood_allsamples = neglikelihood_z_meaned.mean(1).sum(-1)
        else:
            neglikelihood_allsamples = neglikelihood_z_meaned.sum(1).sum(-1)

        # KL term
        q_z = torch.distributions.Normal(z_mean, z_std)
        p_z = torch.distributions.Normal(0, 1)
        kl = torch.distributions.kl_divergence(q_z, p_z)
        kl_dim = kl.sum(2)
        if self.mean_loss_across_samples:
            kl_allsamples = kl_dim.mean(1)
        else:
            kl_allsamples = kl_dim.sum(1)
        return neglikelihood_allsamples.mean(), kl_allsamples.mean()
