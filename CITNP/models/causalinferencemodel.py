"""
File containing the CausalInferenceModel class
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from CITNP.models.causaltransformercomponents import (
    CausalInfEncoder,
    CausalTransformerDecoderLayer,
)
from CITNP.models.nncomponents import MoGPredictor, Predictor
from CITNP.utils.configs import CausalInfModelConfig, LocalLatentConfig
from CITNP.utils.loss import BaseLoss, CNPLoss, NLLLoss, MixtureGaussianLoss
from CITNP.utils.outputs import ModelOutput
from CITNP.utils.sampler import TrainZSampler
from CITNP.utils.utils import (
    gather_by_node,
    reshape_back_node_attention,
    reshape_for_node_attention,
    set_all_except_batchindex_to_zero,
)


class BaseCausalInference(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.emb_depth = config.emb_depth
        self.dim_feedforward = config.dim_feedforward
        self.nhead = config.nhead
        self.dropout = config.dropout
        self.num_layers_encoder = config.num_layers_encoder
        self.num_nodes = config.num_nodes
        self.sample_attn_mode = config.sample_attn_mode
        self.linear_attention = config.linear_attention
        self.mean_loss_across_samples = config.mean_loss_across_samples
        self.device = config.device
        self.dtype = config.dtype

        self.encoder = CausalInfEncoder(
            d_model=self.d_model,
            emb_depth=self.emb_depth,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            num_layers=self.num_layers_encoder,
            num_nodes=self.num_nodes,
            device=self.device,
            dtype=self.dtype,
            sample_attn_mode=self.sample_attn_mode,
            linear_attention=self.linear_attention,
        )

    def _mask_target_data(
        self,
        target_data: torch.Tensor,
        intervention_index: torch.Tensor,
    ):
        """
        Set all values in target_data to 0 except for the indices specified
        in intervention_index.
        """
        masked_target_data = set_all_except_batchindex_to_zero(
            data=target_data, batch_index=intervention_index
        )
        return masked_target_data

    def _extract_outcome_representation(
        self,
        representation: torch.Tensor,
        outcome_index: torch.Tensor,
        num_trgt: int,
    ):
        encoding = representation[:, -num_trgt:]
        outcome_encoding = gather_by_node(encoding, outcome_index)
        return outcome_encoding, encoding

    def _predict(self, outcome_encoding):
        return self.predictor(outcome_encoding)

    @abstractmethod
    def calculate_loss(
        self,
        model_output: ModelOutput,
        target: torch.Tensor,
        outcome_index: torch.Tensor,
    ):
        """
        Calculate the loss for the model.
        """
        raise NotImplementedError("Loss calculation not implemented")

    @abstractmethod
    def forward(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        intervention_index: torch.Tensor,
        outcome_index: torch.Tensor,
        variable_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the CausalInferenceModel class.
        """
        raise NotImplementedError("Forward pass not implemented")


class CausalInferenceModel(BaseCausalInference):
    """
    Class containing the CausalInferenceModel class.
    """

    def __init__(
        self,
        config: CausalInfModelConfig,
    ):
        """
        Constructor for the CausalInferenceModel class
        """
        super().__init__(config)

        self.loss: BaseLoss = CNPLoss(
            mean_loss_across_samples=self.mean_loss_across_samples,
            reduce="mean",
        )
        self.eval_loss: BaseLoss = CNPLoss(
            mean_loss_across_samples=self.mean_loss_across_samples,
            reduce="sum",
        )

        self.predictor = Predictor(
            d_model=self.d_model,
            depth=self.emb_depth,
        )
        assert self.predictor.output_heads == (
            "mean",
            "std",
        ), "CNPLoss requires predictor to output mean and std"

    def calculate_loss(
        self,
        model_output: ModelOutput,
        target: torch.tensor,
        outcome_index: torch.Tensor,
    ):
        """
        Calculate the loss for the model.
        """
        # Get the outcome
        target = gather_by_node(target, outcome_index)
        if self.training:
            return self.loss.calculate_loss(
                model_output=model_output, trgt_outcome=target
            )
        else:
            return self.eval_loss.calculate_loss(
                model_output=model_output, trgt_outcome=target
            )

    def forward(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        intervention_index: torch.Tensor,
        outcome_index: torch.Tensor,
        variable_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the CausalInferenceModel class.

        Finds the distribution p(x_{i} | do(x_{k})) for nodes i in the graph.
        The context data is the observational data of all the nodes.
        The target data contains values of x_k to intervene on. Nodes that
        are not intervened on are set to 0. The intervention index is the index
        of the node that is intervened on.
        The outcome index is the index of the node that we want to find the
        distribution for (x_i).

        Args:
        -----
        - context_data (torch.Tensor): Context data tensor with shape
            (batch_size, num_context, num_nodes, 1)
        - target_data (torch.Tensor): Target data tensor with shape
            (batch_size, num_target, num_nodes, 1)
        - intervention_index (torch.Tensor): Intervention index tensor with shape
            (batch_size)
        - outcome_index (torch.Tensor): Outcome index tensor with shape
            (batch_size)
        """
        masked_target_data = self._mask_target_data(
            target_data=target_data, intervention_index=intervention_index
        )
        # shape rep: (batch_size, num_target, num_nodes, d_model)
        rep, num_trgt = self.encoder.encode(
            context_data=context_data,
            target_data=masked_target_data,
            target_train_data=None,
            outcome_indices=outcome_index,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
        )
        # shape outcome_encoding: (batch_size, num_target, d_model)
        outcome_encoding, _ = self._extract_outcome_representation(
            representation=rep,
            outcome_index=outcome_index,
            num_trgt=num_trgt,
        )
        if self.predictor.output_heads == ("mean", "std"):
            mean, std = self._predict(outcome_encoding)
            return ModelOutput(pred_mean=mean, pred_std=std)
        elif self.predictor.output_heads == ("mean", "std", "weights"):
            mean, std, weights = self._predict(outcome_encoding)
            return ModelOutput(pred_mean=mean, pred_std=std, weights=weights)


class MoGCausalInferenceModel(CausalInferenceModel):
    def __init__(self, config: CausalInfModelConfig):
        super().__init__(config)

        self.loss = MixtureGaussianLoss(
            mean_loss_across_samples=self.mean_loss_across_samples,
            reduce="mean",
            num_mixture_components=config.num_mixture_components,
        )
        self.eval_loss = MixtureGaussianLoss(
            mean_loss_across_samples=self.mean_loss_across_samples,
            reduce="sum",
            num_mixture_components=config.num_mixture_components,
        )

        self.predictor = MoGPredictor(
            d_model=self.d_model,
            depth=self.emb_depth,
            num_mixture_components=config.num_mixture_components,
        )
        assert self.predictor.output_heads == ("mean", "std", "weights"), (
            "MixtureGaussianLoss requires predictor to output "
            "mean, std and weights"
        )


class LocalLatentCausalInferenceModel(BaseCausalInference):
    def __init__(
        self,
        config: LocalLatentConfig,
    ):
        super().__init__(config)
        self.num_z_samples = config.num_z_samples_train
        self.num_z_samples_eval = config.num_z_samples_eval
        self.mean_loss_across_samples = config.mean_loss_across_samples

        self.decoder_depth = config.decoder_depth

        if self.decoder_depth != 0:
            self.decoder = nn.TransformerDecoder(
                CausalTransformerDecoderLayer(
                    d_model=config.d_model,
                    nhead=config.nhead,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    norm_first=True,
                    batch_first=True,
                    device=config.device,
                    dtype=config.dtype,
                    bias=True,
                ),
                num_layers=config.decoder_depth,
            )

        self.train_z_sampler = TrainZSampler(
            d_model=self.d_model,
            nhead=self.nhead,
            emb_depth=self.emb_depth,
            device=self.device,
            dtype=self.dtype,
        )

        self.loss = NLLLoss(
            mean_loss_across_samples=self.mean_loss_across_samples, reduce="mean"
        )
        self.eval_loss = NLLLoss(
            mean_loss_across_samples=self.mean_loss_across_samples, reduce="sum"
        )

        self.predictor = Predictor(
            d_model=self.d_model,
            depth=self.emb_depth,
        )

    def _sample_z(
        self, outcome_rep: torch.Tensor, trgt_rep: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        -----
        - outcome_rep (torch.Tensor): Tensor of shape
            (batch_size, num_trgt, d_model) containing the representation of the
            outcome node.
        - trgt_rep (torch.Tensor): Tensor of shape
            (batch_size, num_trgt, num_nodes, d_model) containing the
            representation of the target nodes.

        Returns:
        --------
        - merged_rep (torch.Tensor): Tensor of shape
            (num_z_samples, batch_size, num_trgt, d_model) containing the
            merged representation of the outcome and z samples.
        """
        num_samples = (
            self.num_z_samples if self.training else self.num_z_samples_eval
        )
        merged_rep, _, _ = self.train_z_sampler(
            outcome_rep=outcome_rep, trgt_rep=trgt_rep, num_samples=num_samples
        )
        return merged_rep

    def _decode(
        self,
        representation: torch.Tensor,
    ):
        """
        Args:
        -----
        - representation (torch.Tensor): Tensor of shape
            (num_z_samples, batch_size, num_trgt, d_model) containing the
            merged representation of the outcome and z samples.

        Returns:
        --------
        - decoder_rep (torch.Tensor): Tensor of shape
            (num_z_samples, batch_size, num_trgt, d_model) containing the
            decoded representation.
        """
        if self.decoder_depth == 0:
            decoder_rep = representation
        else:
            num_z_samples, batch_size, num_trgt, d_model = representation.shape
            rep_batched = reshape_for_node_attention(
                representation,
                num_z_samples,
                batch_size,
                num_trgt,
                d_model,
            )
            decoder_rep_batched = self.decoder(
                tgt=rep_batched,
                memory=None,
            )
            decoder_rep = reshape_back_node_attention(
                decoder_rep_batched,
                num_z_samples,
                batch_size,
                num_trgt,
                d_model,
            )
        return decoder_rep

    def calculate_loss(
        self,
        model_output: ModelOutput,
        target: torch.Tensor,
        outcome_index: torch.Tensor,
    ):
        trgt_outcome = gather_by_node(target, outcome_index)
        if self.training:
            return self.loss.calculate_loss(
                model_output=model_output, trgt_outcome=trgt_outcome
            )
        else:
            return self.eval_loss.calculate_loss(
                model_output=model_output, trgt_outcome=trgt_outcome
            )

    def forward(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        intervention_index: torch.Tensor,
        outcome_index: torch.Tensor,
        variable_mask: Optional[torch.Tensor] = None,
    ):
        masked_target_data = self._mask_target_data(
            target_data=target_data, intervention_index=intervention_index
        )
        # shape rep: (batch_size, num_samples, num_nodes, d_model)
        rep, num_trgt = self.encoder.encode(
            context_data=context_data,
            target_data=masked_target_data,
            target_train_data=None,
            outcome_indices=outcome_index,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
        )
        outcome_rep, trgt_rep = self._extract_outcome_representation(
            representation=rep,
            outcome_index=outcome_index,
            num_trgt=num_trgt,
        )
        # shape z_samples: (num_z_samples, batch_size, num_trgt, d_model)
        z_rep_merged = self._sample_z(outcome_rep, trgt_rep)
        decoder_rep = self._decode(z_rep_merged)
        pred_mean, pred_std = self._predict(decoder_rep)
        return ModelOutput(pred_mean=pred_mean, pred_std=pred_std)
