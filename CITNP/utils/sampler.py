from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_mean_mlp(m):
    if isinstance(m, nn.Linear):
        # Small weights: mean 0, very small std
        nn.init.kaiming_normal_(m.weight)
        m.weight.data[:] *= 0.01
        nn.init.constant_(m.bias, 0.0)  # μ ≈ 0


def init_std_mlp(m):
    if isinstance(m, nn.Linear):
        # Small weights: mean 0, very small std
        nn.init.kaiming_normal_(m.weight)
        m.weight.data[:] *= 0.01
        nn.init.constant_(m.bias, 0.5)


class LatentSampler(nn.Module):

    def __init__(self):
        super().__init__()

    def sample(
        self, context: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TrainZSampler(LatentSampler):

    def __init__(self, encoder, d_model, num_samples, device, dtype):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.num_samples = num_samples

        self.z_mean = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.z_std = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        init_mean_mlp(self.z_mean)
        init_std_mlp(self.z_std)

    def sample(self, outcome_yx: torch.Tensor, **kwargs):
        z_enc = self.encoder(outcome_yx)
        mean = self.z_mean(z_enc)
        std = F.softplus(self.z_std(z_enc))

        z_samples = torch.distributions.Normal(mean, std).rsample((self.num_samples,))
        return z_samples, mean, std


class EvalZSampler(LatentSampler):

    def __init__(self, d_model, num_samples_eval):
        super().__init__()
        self.d_model = d_model
        self.num_samples_eval = num_samples_eval

    def sample(self, trgt_data: torch.Tensor, **kwargs):
        batch_size, num_trgt = trgt_data.size(0), trgt_data.size(1)
        z_samples = torch.randn(
            self.num_samples_eval,
            batch_size,
            num_trgt,
            self.d_model,
            device=trgt_data.device,
            dtype=trgt_data.dtype,
        )
        mean = torch.zeros_like(
            z_samples[0], device=trgt_data.device, dtype=trgt_data.dtype
        )
        std = torch.ones_like(
            z_samples[0], device=trgt_data.device, dtype=trgt_data.dtype
        )
        return z_samples, mean, std
