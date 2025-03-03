"""
File containing the CausalInferenceModel class
"""
import torch

from CITNP.models.causaltransformercomponents import (CausalInfEncoder,
                                                      build_mlp)


class CausalInferenceModel:
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
        num_layers_decoder,
        num_nodes,
        device=None,
        dtype=None,
    ):
        """
        Constructor for the CausalInferenceModel class
        """
        self.d_model = d_model
        self.emb_depth = emb_depth
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
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

        self.predictor = build_mlp(
            self.d_model,
            self.d_model,
            dim_out=2, # mean and std
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
        target_data = target_data.clone()
        batch_size, num_target, num_nodes, _ = target_data.shape

        # Create a tensor that lists node indices for broadcasting.
        # Shape: (1, 1, num_nodes, 1)
        node_range = torch.arange(num_nodes, device=target_data.device).view(1, 1, num_nodes, 1)

        # Reshape intervention_index to (batch_size, 1, 1, 1) for broadcasting.
        intervention_index = intervention_index.view(batch_size, 1, 1, 1)

        # Create a mask: for each sample, only the intervention node gets a 1; the rest are 0.
        mask = (node_range == intervention_index).float()  # Shape: (batch_size, 1, num_nodes, 1)

        # Broadcast the mask along the num_target dimension and apply it.
        target_data = target_data * mask

        return target_data

    def append_outcome_index_to_context_and_target(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        outcome_index: torch.Tensor,
    ):
        # Make outcome index one hot
        outcome_index_onehot = torch.zeros(
            outcome_index.size(0), self.num_nodes, device=outcome_index.device
        )
        outcome_index_onehot.scatter_(1, outcome_index.unsqueeze(1), 1)
        # shape of outcome_index_onehot: (batch_size, 1, num_nodes, 1)
        outcome_index_onehot = outcome_index_onehot.unsqueeze(1).unsqueeze(-1)
        context_data = torch.cat([context_data, outcome_index_onehot], dim=1)
        target_data = torch.cat([target_data, outcome_index_onehot], dim=1)
        return context_data, target_data

    def forward(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        intervention_index: torch.Tensor,
        outcome_index: torch.Tensor,
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
        # Shape of context_data: (batch_size, num_context + 1, num_nodes, 1)
        # Shape of masked_target_data: (batch_size, num_target + 1, num_nodes, 1)
        context_data, masked_target_data = self.append_outcome_index_to_context_and_target(
            context_data, masked_target_data, outcome_index
        )
        # shape of encoder_output: (batch_size, num_target + 1, num_nodes, d_model)
        encoding = self.encoder(context_data, masked_target_data)

        intervention_encoding = torch.gather(
            encoding, 2, intervention_index
        )
        pred = self.predictor(intervention_encoding)
        mean, std = pred.split(1, dim=-1)
        std = torch.nn.functional.softplus(std)
        return mean, std
