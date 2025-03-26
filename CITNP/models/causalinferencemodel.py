"""
File containing the CausalInferenceModel class
"""

from typing import Optional

import torch
import torch.nn as nn

from CITNP.models.causaltransformercomponents import (
    MHCA,
    CausalInfEncoder,
    CausalTransformerDecoderLayer,
    build_mlp,
)
from CITNP.utils.configs import CausalInfModel, LocalLatentConfig
from CITNP.utils.loss import NLLLoss, VILoss
from CITNP.utils.sampler import EvalZSampler, TrainZSampler
from CITNP.utils.utils import (
    append_treatment_outcome_index_to_data,
    gather_by_node,
    set_all_except_batchindex_to_zero,
)


class CausalInferenceModel(torch.nn.Module):
    """
    Class containing the CausalInferenceModel class.
    """

    def __init__(
        self,
        config: CausalInfModel,
    ):
        """
        Constructor for the CausalInferenceModel class
        """
        super(CausalInferenceModel, self).__init__()
        self.d_model = config.d_model
        self.emb_depth = config.emb_depth
        self.dim_feedforward = config.dim_feedforward
        self.nhead = config.nhead
        self.dropout = config.dropout
        self.num_layers_encoder = config.num_layers_encoder
        self.num_nodes = config.num_nodes
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
        )

        self.predictor = build_mlp(
            self.d_model,
            self.d_model,
            dim_out=2,  # mean and std
            depth=2,
        )

    def calculate_loss(self, pred_mean, pred_std, target, outcome_index, **kwargs):
        """
        Calculate the loss for the model.
        """
        # Get the outcome
        target = gather_by_node(target, outcome_index)

        # Calculate the loss
        # TODO (cristiana): Check with Anish whether 1e-30 is for stability.
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)
        loss = -distribution.log_prob(target)  # + 1e-30)
        # Mean over the samples and mean over the batch
        return loss.mean(1).mean()

    def encode_inputs(
        self, context_data, target_data, intervention_index, outcome_index
    ):
        # Shape context_data: (batch_size, num_context + 1, num_nodes, 1)
        # Shape masked_target_data: (batch_size, num_target + 1, num_nodes, 1)
        encoded_cntxt = append_treatment_outcome_index_to_data(
            data=context_data,
            outcome_index=outcome_index,
            treatment_index=intervention_index,
        )
        encoded_trgt = append_treatment_outcome_index_to_data(
            data=target_data,
            outcome_index=outcome_index,
            treatment_index=intervention_index,
        )
        return encoded_cntxt, encoded_trgt

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
        masked_target_data = set_all_except_batchindex_to_zero(
            data=target_data, batch_index=intervention_index
        )

        encoded_cntxt, encoded_trgt = self.encode_inputs(
            context_data, masked_target_data, intervention_index, outcome_index
        )

        # shape encoding: (batch_size, num_target + 1, num_nodes, d_model)
        rep, num_trgt = self.encoder.encode(
            encoded_cntxt,
            encoded_trgt,
            target_train_data=None,
            outcome_indices=outcome_index,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
        )
        encoding = rep[:, -(num_trgt + 1) : -1]

        # We want to find p(outcome | do(intervention))
        outcome_encoding = gather_by_node(encoding, outcome_index)

        pred = self.predictor(outcome_encoding)
        mean, std = pred.split(1, dim=-1)
        std = torch.nn.functional.softplus(std)

        return mean, std, None, None


class LocalLatentCausalInferenceModel(CausalInferenceModel):

    def __init__(
        self,
        config: LocalLatentConfig,
    ):
        causalconfig = CausalInfModel(
            d_model=config.d_model,
            emb_depth=config.emb_depth,
            dim_feedforward=config.dim_feedforward,
            nhead=config.nhead,
            dropout=config.dropout,
            num_layers_encoder=config.num_layers_encoder,
            num_nodes=config.num_nodes,
            device=config.device,
            dtype=config.dtype,
        )
        super().__init__(causalconfig)
        self.num_z_samples = config.num_z_samples_train
        self.num_z_samples_eval = config.num_z_samples_eval
        self.use_vi = config.use_vi

        self.decoder_depth = config.decoder_depth

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
                bias=False,
            ),
            num_layers=config.decoder_depth,
        )

        self.z_param_mlp = build_mlp(
            self.d_model,
            self.d_model,
            dim_out=2 * config.d_model,  # mean and std
            depth=2,
        )

        self.z_rep_merger = MHCA(
            d_model=self.d_model,
            nhead=config.nhead,
            dim_feedforward=self.dim_feedforward,
            norm_first=True,
            batch_first=True,
            device=self.device,
            dtype=self.dtype,
            bias=False,
        )

        self.train_z_sampler = TrainZSampler(
            self.z_param_mlp, self.d_model, self.num_z_samples
        )
        self.eval_z_sampler = EvalZSampler(self.d_model, self.num_z_samples_eval)

        self.loss = NLLLoss() if not self.use_vi else VILoss()
        self.eval_loss = NLLLoss()

    def extract_training_target(self, target_data, intervention_index, outcome_index):
        if self.training:
            # During training we want the true outcome.
            only_intvn_trgt = set_all_except_batchindex_to_zero(
                data=target_data, batch_index=intervention_index
            )
            only_outcome_trgt = set_all_except_batchindex_to_zero(
                data=target_data, batch_index=outcome_index
            )
            masked_trgt_train = only_intvn_trgt + only_outcome_trgt
        else:
            masked_trgt_train = None
        return masked_trgt_train

    def extract_representation(
        self,
        representation,
        num_trgt,
    ):
        """
        Extract the representation based on training or not.

        Returns:
        --------
        - R_x (torch.Tensor): Tensor of shape (batch_size, num_trgt, d_model)
            containing the representation of the target nodes.
        - R_yx (torch.Tensor): Tensor of shape (batch_size, num_trgt, d_model)
            containing the representation of the target nodes with the outcome node.
        """
        R_x = representation[:, -(num_trgt + 1) : -1]
        if self.training:
            start_index = -(2 * num_trgt + 1)
            end_index = start_index + num_trgt
            R_yx = representation[:, start_index:end_index]
        else:
            R_yx = None
        return R_x, R_yx

    def sample_z(self, outcome_yx, trgt_data):
        """
        Args:
        -----
        - outcome_yx (torch.Tensor): Tensor of shape (batch_size, num_trgt, d_model)
            containing the parameters of the distribution of z.

        Returns:
        --------
        - z_samples (torch.Tensor): Tensor of shape (num_z_samples, batch_size, d_model)
        """
        # shape z: (batch_size, num_trgt, 1, 2*d_model)
        if self.training:
            zsamples, mean, std = self.train_z_sampler.sample(
                outcome_yx, num_samples=self.num_z_samples
            )
        else:
            zsamples, mean, std = self.eval_z_sampler.sample(
                trgt_data, num_samples=self.num_z_samples_eval
            )

        return zsamples, mean, std

    def merge_zsamples_rep(self, z_samples, rep):
        """
        Args:
        -----
        - z_samples (torch.Tensor): Tensor of shape
            (num_z_samples, batch_size, num_trgt, d_model)
        - rep (torch.Tensor): Tensor of shape
            (batch_size, num_trgt, d_model)
        """
        num_z_samples, batch_size, num_trgt, d_model = z_samples.shape
        # shape rep_expand: (num_z_samples, batch_size, num_trgt, d_model)
        rep_expand = rep.unsqueeze(0).expand_as(z_samples)

        # shape rep_batched: (num_z_samples * batch_size, num_trgt, d_model)
        rep_batched = rep_expand.reshape(num_z_samples * batch_size, num_trgt, d_model)
        z_batched = z_samples.reshape(num_z_samples * batch_size, num_trgt, d_model)

        # Cross attention
        z_rep = self.z_rep_merger(
            query=rep_batched,
            key=z_batched,
            value=z_batched,
        )
        z_rep_unbatched = z_rep.reshape(num_z_samples, batch_size, num_trgt, d_model)
        return z_rep_unbatched

    def forward(
        self,
        context_data,
        target_data,
        intervention_index,
        outcome_index,
        variable_mask=None,
    ):
        masked_trgt_train = self.extract_training_target(
            target_data, intervention_index, outcome_index
        )

        masked_target_data = set_all_except_batchindex_to_zero(
            data=target_data, batch_index=intervention_index
        )

        encoded_ctnxt, encoded_trgt = self.encode_inputs(
            context_data, masked_target_data, intervention_index, outcome_index
        )

        # shape rep: (batch_size, num_samples + 2, num_nodes, d_model)
        rep, num_trgt = self.encoder.encode(
            context_data=encoded_ctnxt,
            target_data=encoded_trgt,
            target_train_data=masked_trgt_train,
            outcome_indices=outcome_index,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
        )
        # shape encoding: (batch_size, num_trgt + 1, num_nodes, d_model)
        R_x, R_yx = self.extract_representation(rep, num_trgt)

        # We want to find p(outcome | do(intervention))
        # shape outcome_encoding: (batch_size, num_trgt, d_model)
        outcome_x = gather_by_node(R_x, outcome_index)
        if self.training:
            outcome_yx = gather_by_node(R_yx, outcome_index)
        else:
            outcome_yx = None

        # shape z_samples: (num_z_samples, batch_size, num_trgt, d_model)
        z_samples, z_mean, z_std = self.sample_z(outcome_yx, masked_target_data)

        # Merge outcome_rep and z_samples
        final_rep = self.merge_zsamples_rep(z_samples, outcome_x)

        if self.decoder_depth != 0:
            num_z_samples, batch_size, num_trgt, _ = final_rep.shape
            final_rep_batched = final_rep.reshape(
                num_z_samples * batch_size, num_trgt, -1
            )
            decoder_rep_batched = self.decoder(
                tgt=final_rep_batched,
                memory=None,
            )
            decoder_rep = decoder_rep_batched.reshape(
                num_z_samples, batch_size, num_trgt, -1
            )
        else:
            decoder_rep = final_rep

        pred = self.predictor(decoder_rep)
        mean, std = pred.split(1, dim=-1)
        std = torch.nn.functional.softplus(std)

        return mean, std, z_mean, z_std

    def calculate_loss(self, pred_mean, pred_std, target, outcome_index, z_mean, z_std):
        """
        Args:
        -----
        - pred_mean (torch.Tensor): Tensor of shape
            (num_z_samples, batch_size, num_trgt, 1) containing the mean of the
            predicted distribution.
        - pred_std (torch.Tensor): Tensor of shape
            (num_z_samples, batch_size, num_trgt, 1) containing the std of the
            predicted distribution.
        - target (torch.Tensor): Tensor of shape
            (batch_size, num_trgt, num_nodes, 1) containing the target values.
        - outcome_index (torch.Tensor): Tensor of shape (batch_size,) with the index
            of the outcome node for each dataset.
        - z_mean (torch.Tensor): Tensor of shape
            (batch_size, num_trgt, 1, d_model) containing the mean of the
            z distribution.
        - z_std (torch.Tensor): Tensor of shape
            (batch_size, num_trgt, 1, d_model) containing the std of the
            z distribution.
        """
        trgt_outcome = gather_by_node(target, outcome_index)
        if self.training:
            return self.loss.calculate_loss(
                pred_mean=pred_mean,
                pred_std=pred_std,
                target=trgt_outcome,
                z_mean=z_mean,
                z_std=z_std,
            )
        else:
            return self.eval_loss.calculate_loss(
                pred_mean=pred_mean,
                pred_std=pred_std,
                target=trgt_outcome,
                z_mean=z_mean,
                z_std=z_std,
            )
