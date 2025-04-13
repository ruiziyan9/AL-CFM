from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CITNP.models.nncomponents import FlashMHCA, build_residual_network
from CITNP.utils.utils import (
    reshape_back_node_attention,
    reshape_for_node_attention,
)


class LatentSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def _sample(
        self, rep: torch.Tensor, num_samples: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TrainZSampler(LatentSampler):
    def __init__(self, d_model, nhead, emb_depth, device, dtype):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.emb_depth = emb_depth
        self.device = device
        self.dtype = dtype

        self.pre_z_encoder = FlashMHCA(
            d_model=self.d_model,
            nhead=self.nhead,
            batch_first=True,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.z_encoder = build_residual_network(
            num_blocks=1,
            input_dim=self.d_model,
            hidden_dim=self.d_model,
            output_dim=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.z_rep_merger = build_residual_network(
            num_blocks=self.emb_depth,
            input_dim=self.d_model * 2,
            hidden_dim=self.d_model,
            output_dim=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )

        self.z_mean = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.z_std = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def _prepare_inputs_for_merger(
        self,
        sample_outcome_rep: torch.Tensor,
        z_samples: torch.Tensor,
    ) -> torch.Tensor:
        num_z_samples, batch_size, num_trgt, d_model = z_samples.shape
        # shape rep_batched: (num_z_samples * batch_size, num_trgt, d_model)
        rep_batched = sample_outcome_rep.reshape(
            num_z_samples * batch_size, num_trgt, d_model
        )
        z_batched = z_samples.reshape(
            num_z_samples * batch_size, num_trgt, d_model
        )
        # concatenate along the last dimension
        z_rep_batched = torch.cat((z_batched, rep_batched), dim=-1)
        return z_rep_batched

    def _merge_rep(
        self, outcome_rep: torch.Tensor, z_samples: torch.Tensor
    ) -> torch.Tensor:
        num_z_samples, batch_size, num_trgt, d_model = z_samples.shape
        sample_outcome_rep = outcome_rep.unsqueeze(0).expand_as(z_samples)
        rep_both = self._prepare_inputs_for_merger(
            sample_outcome_rep=sample_outcome_rep, z_samples=z_samples
        )
        merged_rep = self.z_rep_merger(rep_both)
        # shape merged_rep: (num_z_samples * batch_size, num_trgt, d_model)
        unbatch_merged_rep = merged_rep.reshape(
            num_z_samples, batch_size, num_trgt, d_model
        )
        return unbatch_merged_rep

    def _sample(
        self, rep: torch.Tensor, num_samples: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_enc = self.z_encoder(rep)
        mean = self.z_mean(z_enc)
        std = F.softplus(self.z_std(z_enc))
        z_samples = torch.distributions.Normal(mean, std).rsample((num_samples,))
        return z_samples, mean, std

    def _encode_prez(
        self, outcome_rep: torch.Tensor, trgt_rep: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_trgt, num_nodes, d_model = trgt_rep.shape
        outcome_rep = outcome_rep.unsqueeze(2)

        batched_outcome_rep = reshape_for_node_attention(
            outcome_rep, batch_size, num_trgt, 1, d_model
        )
        batched_trgt_rep = reshape_for_node_attention(
            trgt_rep, batch_size, num_trgt, num_nodes, d_model
        )

        pre_z_rep = self.pre_z_encoder(
            query=batched_outcome_rep,
            key=batched_trgt_rep,
            value=batched_trgt_rep,
        )
        # shape pre_z_rep: (batch_size, num_trgt, 1, d_model)
        unbatched_pre_z_rep = reshape_back_node_attention(
            pre_z_rep, batch_size, num_trgt, 1, d_model
        )
        unbatched_pre_z_rep = unbatched_pre_z_rep.squeeze(2)
        return unbatched_pre_z_rep

    def forward(
        self, outcome_rep: torch.Tensor, trgt_rep: torch.Tensor, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
        -----
        - outcome_rep: (batch_size, num_trgt, d_model)
        - trgt_rep: (batch_size, num_trgt, num_nodes, d_model)
        - num_samples: number of samples to draw from the distribution

        Returns:
        -------
        - merged_rep: (num_samples, batch_size, num_trgt, d_model)
        - mean: (batch_size, num_trgt, d_model)
        - std: (batch_size, num_trgt, d_model)
        """
        # shape pre_z_rep: (batch_size, num_trgt, d_model)
        pre_z_rep = self._encode_prez(outcome_rep, trgt_rep)
        z_samples, mean, std = self._sample(pre_z_rep, num_samples)
        merged_rep = self._merge_rep(outcome_rep, z_samples)
        return merged_rep, mean, std
