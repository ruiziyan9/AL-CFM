"""
File containing the CausalInferenceModel class
"""

from typing import Optional

import torch

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
        encoding = torch.zeros(
            batch_size, self.num_nodes, device=outcome_index.device
        )

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

    def move_outcome_to_first_node_and_encode_treatment(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        outcome_index: torch.Tensor,
        treatment_index: torch.Tensor,
    ):
        """
        Encode treatment with one hot and move the outcome to the first element
        of the node axis.

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
        # Encode the treatment index with one hot encoding with -1 for
        # the treatment index and 0 for the rest of the nodes
        encoding = torch.zeros(
            batch_size, self.num_nodes, device=outcome_index.device
        )
        # Set treatment index to -1 for each batch sample
        encoding[torch.arange(batch_size), treatment_index] = -1
        # Reshape to (batch_size, 1, num_nodes, 1) for concatenation
        encoding = encoding.unsqueeze(1).unsqueeze(-1)
        # Concatenate the single encoding tensor along the channel dimension
        context_data = torch.cat([context_data, encoding], dim=1)
        target_data = torch.cat([target_data, encoding], dim=1)

        # Move the outcome index to the first node
        # Build new indices that move the outcome node to the first position.
        # Create a tensor for node indices [0, 1, ..., num_nodes - 1] for each batch.
        nodes = (
            torch.arange(self.num_nodes, device=outcome_index.device)
            .unsqueeze(0)
            .expand(batch_size, self.num_nodes)
        )
        # For each batch sample, create a mask that is True for nodes that are NOT the outcome.
        mask = nodes != outcome_index.unsqueeze(1)
        # Extract the other node indices and reshape to (batch_size, num_nodes - 1)
        other_nodes = nodes[mask].view(batch_size, self.num_nodes - 1)
        # Concatenate the outcome index as the first element with the rest of the indices.
        new_indices = torch.cat(
            [outcome_index.unsqueeze(1), other_nodes], dim=1
        )  # shape: (batch_size, num_nodes)
        # Expand indices for gathering along the node dimension.
        # For context_data: shape is (batch_size, samples, num_nodes, 1)
        new_indices_exp = (
            new_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(batch_size, context_data.shape[1], self.num_nodes, 1)
        )
        context_data = torch.gather(context_data, dim=2, index=new_indices_exp)
        # Do the same for target_data.
        new_indices_exp = (
            new_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(batch_size, target_data.shape[1], self.num_nodes, 1)
        )
        target_data = torch.gather(target_data, dim=2, index=new_indices_exp)

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
        # Shape of context_data: (batch_size, num_context + 1, num_nodes, 1)
        # Shape of masked_target_data: (batch_size, num_target + 1, num_nodes, 1)
        context_data, masked_target_data = (
            self.append_treatment_outcome_index_to_context_and_target(
                context_data,
                masked_target_data,
                outcome_index,
                treatment_index=intervention_index,
            )
        )
        outcome_indices = outcome_index

        # shape of encoder_output: (batch_size, num_target + 1, num_nodes, d_model)
        encoding = self.encoder.encode(
            context_data,
            masked_target_data,
            outcome_indices=outcome_indices,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
        )

        # We want to find p(outcome | do(intervention))
        batch_idx = torch.arange(encoding.size(0), device=encoding.device)
        outcome_encoding = encoding[batch_idx, :, outcome_indices, :]

        pred = self.predictor(outcome_encoding)
        mean, std = pred.split(1, dim=-1)
        std = torch.nn.functional.softplus(std)

        # Get rid of the extra sample for the outcome index
        mean = mean[:, :-1]
        std = std[:, :-1]
        return mean, std
