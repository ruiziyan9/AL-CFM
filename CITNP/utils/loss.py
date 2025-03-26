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

    def __init__(self):
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
        # During evaluation we only care about the predictive likelihood
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)
        neglikelihood = -distribution.log_prob(target)
        neglikelihood_sample_mean = neglikelihood.mean(2)
        neglikelihood_z_meaned = torch.logsumexp(
            neglikelihood_sample_mean, dim=0
        ) - torch.log(
            torch.tensor(
                self.num_z_samples_eval,
                device=neglikelihood.device,
                dtype=neglikelihood.dtype,
            )
        )
        neglikelihood_fullmeaned = neglikelihood_z_meaned.mean(0).mean()
        return neglikelihood_fullmeaned


class VILoss(BaseLoss):

    def __init__(self):
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
        neglikelihood_fullmeaned = neglikelihood_z_meaned.mean(1).mean()

        # KL term
        q_z = torch.distributions.Normal(z_mean, z_std)
        p_z = torch.distributions.Normal(0, 1)
        kl = torch.distributions.kl_divergence(q_z, p_z)
        kl_meaned = kl.mean(-1).mean(1).mean(0)
        return neglikelihood_fullmeaned + kl_meaned
