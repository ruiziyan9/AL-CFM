"""
File containing the CausalInferenceModel class
"""

from typing import Optional

import torch
import torch.nn as nn

from CITNP.models.causaltransformercomponents import CausalInfEncoder, build_mlp


class CausalInferenceModel(torch.nn.Module):
    """
    Class containing the CausalInferenceModel class.
    """

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_nodes,
        device=None,
        dtype=None,
    ):
        """
        Constructor for the CausalInferenceModel class
        """
        super(CausalInferenceModel, self).__init__()
        self.d_model = d_model
        self.emb_depth = emb_depth
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers_encoder = num_layers_encoder
        self.num_nodes = num_nodes
        self.device = device
        self.dtype = dtype

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

        # self.decoder = nn.TransformerDecoder(
        #     CausalTransformerDecoderLayer(
        #         d_model=d_model,
        #         nhead=nhead,
        #         dim_feedforward=dim_feedforward,
        #         dropout=dropout,
        #         norm_first=True,
        #         batch_first=True,
        #         device=device,
        #         dtype=dtype,
        #         bias=False,
        #     ),
        #     num_layers=2
        # )

        self.predictor = build_mlp(
            self.d_model,
            self.d_model,
            dim_out=2,  # mean and std
            depth=2,
        )

    def set_nonintervention_targets_to_zero(
        self,
        target_data: torch.Tensor,
        intervention_index: torch.Tensor,
    ):
        """
        Sets all target values corresponding to non-intervention nodes to zero.

        Args:
        -----
        - target_data (torch.Tensor): Tensor of shape
            (batch_size, num_target, num_nodes, 1) containing target values.
        - intervention_index (torch.Tensor): Tensor of shape (batch_size,) with the index
             of the intervention node for each sample.

        Returns:
        --------
            torch.Tensor: Modified target_data with non-intervention targets set to zero.
        """
        device = target_data.device
        dtype = target_data.dtype
        target_data = target_data.clone().to(device)
        batch_size, num_target, num_nodes, _ = target_data.shape

        # Create a tensor that lists node indices for broadcasting.
        # Shape: (1, 1, num_nodes, 1)
        node_range = torch.arange(num_nodes, device=target_data.device).view(
            1, 1, num_nodes, 1
        )

        # Reshape intervention_index to (batch_size, 1, 1, 1) for broadcasting.
        intervention_index = intervention_index.view(batch_size, 1, 1, 1)

        # Create a mask: for each sample, only the intervention node gets a 1; the rest are 0.
        mask = (
            node_range == intervention_index
        ).float()  # Shape: (batch_size, 1, num_nodes, 1)

        # Broadcast the mask along the num_target dimension and apply it.
        target_data = target_data * mask
        target_data = target_data.to(dtype)
        return target_data

    def append_treatment_outcome_index_to_context_and_target(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        outcome_index: torch.Tensor,
        treatment_index: torch.Tensor,
    ):
        """
        Appends the treatment and outcome index to the context and target data.
        The outcome is encoded by the value 1 and the treatment is encoded by
        the value -1. The rest of the nodes are encoded by 0.

        Args:
        -----
        - context_data (torch.Tensor): Context data tensor with shape
            (batch_size, num_context, num_nodes, 1)
        - target_data (torch.Tensor): Target data tensor with shape
            (batch_size, num_target, num_nodes, 1)
        - outcome_index (torch.Tensor): Outcome index tensor with shape
            (batch_size)
        - treatment_index (torch.Tensor): Treatment index tensor with shape
            (batch_size)
        """
        dtype = context_data.dtype
        batch_size = outcome_index.size(0)

        # Create a tensor of zeros with shape (batch_size, num_nodes)
        encoding = torch.zeros(batch_size, self.num_nodes, device=outcome_index.device)

        # Set outcome index to 1 for each batch sample
        encoding[torch.arange(batch_size), outcome_index] = 1

        # Set treatment index to -1 for each batch sample
        encoding[torch.arange(batch_size), treatment_index] = -1

        # Reshape to (batch_size, 1, num_nodes, 1) for concatenation
        encoding = encoding.unsqueeze(1).unsqueeze(-1)

        # Concatenate the single encoding tensor along the channel dimension
        context_data = torch.cat([context_data, encoding], dim=1)
        target_data = torch.cat([target_data, encoding], dim=1)

        # Convert back to the original dtype if necessary
        context_data = context_data.to(dtype)
        target_data = target_data.to(dtype)

        return context_data, target_data

    def calculate_loss(self, pred_mean, pred_std, target, outcome_index):
        """
        Calculate the loss for the model.
        """
        # Get the outcome
        batch_idx = torch.arange(target.size(0), device=target.device)
        target = target[batch_idx, :, outcome_index, :]

        # Calculate the loss
        # TODO (cristiana): Check with Anish whether 1e-30 is for stability.
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)
        loss = -distribution.log_prob(target)  # + 1e-30)
        # Mean over the samples and mean over the batch
        return loss.mean(1).mean()

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
        masked_target_data = self.set_nonintervention_targets_to_zero(
            target_data, intervention_index
        )
        # Shape context_data: (batch_size, num_context + 1, num_nodes, 1)
        # Shape masked_target_data: (batch_size, num_target + 1, num_nodes, 1)
        context_data, masked_target_data = (
            self.append_treatment_outcome_index_to_context_and_target(
                context_data,
                masked_target_data,
                outcome_index,
                treatment_index=intervention_index,
            )
        )

        # shape encoding: (batch_size, num_target + 1, num_nodes, d_model)
        rep, num_trgt = self.encoder.encode(
            context_data,
            masked_target_data,
            outcome_indices=outcome_index,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
        )
        encoding = rep[:, -num_trgt:]

        # We want to find p(outcome | do(intervention))
        batch_idx = torch.arange(encoding.size(0), device=encoding.device)
        outcome_encoding = encoding[batch_idx, :, outcome_index, :]

        pred = self.predictor(outcome_encoding)
        mean, std = pred.split(1, dim=-1)
        std = torch.nn.functional.softplus(std)

        # Get rid of the extra sample for the outcome index
        mean = mean[:, :-1]
        std = std[:, :-1]
        return mean, std


class LocalLatentCausalInferenceModel(CausalInferenceModel):

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_nodes,
        num_z_samples=10,
        use_vi=True,  # Loss is VI or ML
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers_encoder,
            num_nodes,
            device,
            dtype,
        )
        self.num_z_samples = num_z_samples
        self.use_vi = use_vi

        self.z_param_mlp = build_mlp(
            self.d_model,
            self.d_model,
            dim_out=2 * d_model,  # mean and std
            depth=2,
        )

        self.z_rep_merger = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

    def append_query_vector(self, cntxt_data, trgt_data):
        """
        Appends a query vector of zeros to the data tensor in the sample
        axis. Also appends an indicator variable to the data tensor in the
        node axis.

        Args:
        -----
        - cntxt_data (torch.Tensor): Data tensor with shape
            (batch_size, num_samples, num_nodes, d_model)

        Returns:
        --------
        - query_data (torch.Tensor): Data tensor with shape
            (batch_size, num_samples + 1, num_nodes + 1, d_model)
        """
        dtype = cntxt_data.dtype
        device = cntxt_data.device
        batch_size, num_cntxt, _, d_model = cntxt_data.shape
        _, num_trgt, _, _ = trgt_data.shape

        # Create a tensor of zeros with shape (batch_size, 1, d_model)
        query_vector = torch.zeros(batch_size, 1, d_model, device=device, dtype=dtype)

        # Create a tensor of zeros with shape (batch_size, num_samples + 1, 1)
        indicator_vector = torch.zeros(
            batch_size, num_cntxt + 1, 1, device=device, dtype=dtype
        )
        indicator_vector[:, 0] = 1

        trgt_indicator_vector = torch.zeros(
            batch_size, num_trgt, 1, device=device, dtype=dtype
        )

        # Concatenate the query vector along the sample axis
        query_cntxt = torch.cat([query_vector, cntxt_data], dim=1)

        # Concatenate the indicator vector along the node axis
        query_cntxt = torch.cat([indicator_vector, query_cntxt], dim=2)
        query_trgt = torch.cat([trgt_indicator_vector, trgt_data], dim=2)

        return query_cntxt, query_trgt

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
            z = self.z_param_mlp(outcome_yx)
            mean, std = z.split(self.d_model, dim=-1)
            std = torch.nn.functional.softplus(std)
            z_samples = torch.distributions.Normal(mean, std).rsample(
                (self.num_z_samples,)
            )
        else:
            # Sample from standard normal
            batch_size = trgt_data.size(0)
            num_trgt = trgt_data.size(1) - 1
            z_samples = torch.randn(
                self.num_z_samples,
                batch_size,
                num_trgt,
                self.d_model,
                device=trgt_data.device,
                dtype=trgt_data.dtype,
            )
            mean = torch.zeros(
                batch_size, num_trgt, 1, self.d_model, device=trgt_data.device
            )
            std = torch.ones(
                batch_size, num_trgt, 1, self.d_model, device=trgt_data.device
            )
        return z_samples, mean, std

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
        rep_expand = rep.unsqueeze(0).expand(self.num_z_samples, -1, -1, -1)

        # shape rep_batched: (num_z_samples * batch_size, num_trgt, d_model)
        rep_batched = rep_expand.contiguous().view(
            num_z_samples * batch_size, num_trgt, d_model
        )
        z_batched = z_samples.contiguous().view(
            num_z_samples * batch_size, num_trgt, d_model
        )

        # Cross attention
        z_rep = self.z_rep_merger(
            query=rep_batched,
            key=z_batched,
            value=z_batched,
        )[0]
        z_rep_unbatched = z_rep.view(num_z_samples, batch_size, -1, d_model)
        return z_rep_unbatched

    def only_intervention_outcome(self, target_data, intervention_index, outcome_index):
        """
        Set everything that is not the intervention node and outcome node to zero.

        Args:
        -----
        - target_data (torch.Tensor): Tensor of shape
            (batch_size, num_target, num_nodes, 1) containing target values.
        - intervention_index (torch.Tensor): Tensor of shape (batch_size,) with the index
             of the intervention node for each sample.
        - outcome_index (torch.Tensor): Tensor of shape (batch_size,)
            with the index of the outcome node for each sample.

        Returns:
        --------
        - torch.Tensor: Modified target_data with non-intervention and non-outcome nodes zeroed out.
        """
        batch_size, num_target, num_nodes, _ = target_data.shape

        # Create a mask of shape (batch_size, num_nodes) where True means keep
        keep_mask = torch.zeros(
            (batch_size, num_nodes), dtype=torch.bool, device=target_data.device
        )

        # Mark intervention and outcome nodes as True
        keep_mask.scatter_(1, intervention_index.unsqueeze(1), True)
        keep_mask.scatter_(1, outcome_index.unsqueeze(1), True)

        # Expand the mask to match target_data shape
        # (batch_size, 1, num_nodes, 1) -> broadcast to (batch_size, num_target, num_nodes, 1)
        expanded_mask = keep_mask[:, None, :, None]

        # Zero out everything not in the mask
        target_data = target_data * expanded_mask

        return target_data

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

    def forward(
        self,
        context_data,
        target_data,
        intervention_index,
        outcome_index,
        variable_mask=None,
    ):
        if self.training:
            # During training we want the true outcome.
            masked_intvn_outcome = self.only_intervention_outcome(
                target_data, intervention_index, outcome_index
            )
        else:
            masked_intvn_outcome = None

        masked_target_data = self.set_nonintervention_targets_to_zero(
            target_data, intervention_index
        )
        # Shape context_data: (batch_size, num_context + 1, num_nodes, 1)
        # Shape masked_target_data: (batch_size, num_target + 1, num_nodes, 1)
        context_data, masked_target_data = (
            self.append_treatment_outcome_index_to_context_and_target(
                context_data,
                masked_target_data,
                outcome_index,
                treatment_index=intervention_index,
            )
        )

        # shape rep: (batch_size, num_samples + 2, num_nodes, d_model)
        rep, num_trgt = self.encoder.encode(
            context_data=context_data,
            target_data=masked_target_data,
            target_train_data=masked_intvn_outcome,
            outcome_indices=outcome_index,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
        )
        # shape encoding: (batch_size, num_trgt + 1, num_nodes, d_model)
        R_x, R_yx = self.extract_representation(rep, num_trgt)

        # We want to find p(outcome | do(intervention))
        batch_idx = torch.arange(R_x.size(0), device=R_x.device)
        # shape outcome_encoding: (batch_size, num_trgt, d_model)
        outcome_x = R_x[batch_idx, :, outcome_index, :]
        if self.training:
            outcome_yx = R_yx[batch_idx, :, outcome_index, :]
        else:
            outcome_yx = None

        # shape z_samples: (num_z_samples, batch_size, num_trgt, d_model)
        z_samples, z_mean, z_std = self.sample_z(outcome_yx, masked_target_data)

        # Merge outcome_rep and z_samples
        final_rep = self.merge_zsamples_rep(z_samples, outcome_x)

        pred = self.predictor(final_rep)
        mean, std = pred.split(1, dim=-1)
        std = torch.nn.functional.softplus(std)

        return mean, std, z_mean, z_std

    def vi_loss(self, pred_mean, pred_std, trgt_outcome, z_mean, z_std):
        """
        Calculate the loss for the model.
        """
        # Data term
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)
        neglikelihood = -distribution.log_prob(trgt_outcome)
        neglikelihood_z_meaned = neglikelihood.mean(0)
        neglikelihood_fullmeaned = neglikelihood_z_meaned.mean(1).mean()

        # KL term
        q_z = torch.distributions.Normal(z_mean, z_std)
        p_z = torch.distributions.Normal(0, 1)
        kl = torch.distributions.kl_divergence(q_z, p_z)
        kl_meaned = kl.mean(-1).mean(1).mean(0)
        return neglikelihood_fullmeaned + kl_meaned

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
        batch_idx = torch.arange(target.size(0), device=target.device)
        trgt_outcome = target[batch_idx, :, outcome_index, :]

        if self.use_vi:
            loss = self.vi_loss(pred_mean, pred_std, trgt_outcome, z_mean, z_std)
        else:
            raise NotImplementedError("ML loss not implemented yet.")
        return loss
