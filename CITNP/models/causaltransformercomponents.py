import copy
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def build_mlp(dim_in, dim_hid, dim_out, depth):
    if dim_in == dim_hid:
        modules = [
            nn.Sequential(
                nn.Linear(dim_in, dim_hid, bias=True),
                nn.ReLU(),
            )
        ]
    else:
        modules = [nn.Linear(dim_in, dim_hid, bias=True), nn.ReLU()]
    for _ in range(int(depth) - 2):
        modules.append(
            nn.Sequential(
                nn.Linear(dim_hid, dim_hid, bias=True),
                nn.ReLU(),
            )
        )
    modules.append(nn.Linear(dim_hid, dim_out, bias=True))
    return nn.Sequential(*modules)


class CausalInfTransformerEncoder(nn.Module):
    """
    Causal Inference Transformer that alternates attention between samples and nodes.
    """

    def __init__(
        self,
        encoder_layers: nn.ModuleList,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ) -> None:
        super(CausalInfTransformerEncoder, self).__init__()
        assert len(encoder_layers) > 0, "Encoder must have at least one layer."
        assert len(encoder_layers) % 2 == 0, "Encoder must have an even number of layers."
        self.layers = encoder_layers

    def forward(
        self,
        src: Tensor,
        sample_mask: Tensor,
        variable_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        # src: [batch_size, num_samples, num_nodes, d_model]
        # We need to reshape the tensor to [batch_size * num_nodes, num_samples, d_model]
        # to carry out attention over samples
        batch_size, num_samples, num_nodes, d_model = src.size()
        for idx_layer, mod in enumerate(self.layers):
            # Attention across samples
            if idx_layer % 2 == 0:
                # shape [batch_size, num_nodes, num_samples, d_model]
                src = src.permute(0, 2, 1, 3)
                # shape [batch_size * num_nodes, num_samples, d_model]
                src = src.contiguous().view(batch_size * num_nodes, num_samples, d_model)
                src = mod(src, src_mask=sample_mask, src_key_padding_mask=None, is_causal=is_causal)
                # Reshape the tensor back to [batch_size, num_nodes, num_samples, d_model]
                src = src.view(batch_size, num_nodes, num_samples, d_model)
            # Attention across nodes
            else:
                # shape [batch_size, num_samples, num_nodes, d_model]
                src = src.permute(0, 2, 1, 3)
                # shape [batch_size * num_samples, num_nodes, d_model]
                src = src.contiguous().view(batch_size * num_samples, num_nodes, d_model)
                # Extra zeros for the query
                node_src_key_padding_mask = variable_mask
                if node_src_key_padding_mask is not None:
                    node_src_key_padding_mask = node_src_key_padding_mask.contiguous().view(batch_size * num_samples, num_nodes)
                src = mod(src, src_mask=None, src_key_padding_mask=node_src_key_padding_mask, is_causal=is_causal)
                # Make masking position back to zero
                if node_src_key_padding_mask is not None:
                    bool_pad = node_src_key_padding_mask == -float("inf")
                    src = src.masked_fill_(bool_pad.unsqueeze(-1), 0)
                # Reshape the tensor back to [batch_size, num_samples, num_nodes, d_model]
                src = src.contiguous().view(batch_size, num_samples, num_nodes, d_model)
        return src


class CausalInfEncoder(nn.Module):
    """
    CausalInfEncoder encodes the context and target

    Args:
    -----
    - d_model (int): The dimensionality of the model.
    - dim_feedforward (int): The dimensionality of the feedforward network.
    - nhead (int): The number of attention heads.
    - num_layers (int): The number of transformer encoder layers.
    - num_nodes (int): The number of nodes.
    - device (str): The device to run the module on.
    - dtype (torch.dtype): The data type of the module's parameters.
    - emb_depth (int, optional): The depth of the embedding MLP. Defaults to 2.
    - dropout (float, optional): The dropout rate. Defaults to 0.0.

    Methods:
    --------
    - embed(target_data): Embeds the target data into a d_model dimensional space.
    - compute_summary(query, key, value): Computes the summary representation for the query.
    - encode(target_data): Encodes the target data and computes the summary representation.

    Attributes:
    ----------
    - embedder (nn.Module): The MLP used for embedding.
    - encoder (CausalTransformerEncoder): The CausalTransformerEncoder module.
    - representation (nn.MultiheadAttention): The multi-head attention module.
    - use_positional_encoding (bool): Whether to use positional encoding.
    - positional_encoding (PositionalEncoding): The positional encoding module.
        ...
        - target_data (torch.Tensor): The target data with shape [batch_size, num_samples, num_nodes, 1].
        - embedding (torch.Tensor): The embedded target data with shape [batch_size, num_samples + 1, num_nodes, d_model].
        ...
        - query (torch.Tensor): The query tensor with shape [batch_size, 1, num_nodes, d_model].
        - key (torch.Tensor): The key tensor with shape [batch_size, num_samples, num_nodes, d_model].
        - value (torch.Tensor): The value tensor with shape [batch_size, num_samples, num_nodes, d_model].
        - summary_rep (torch.Tensor): The summary representation with shape [batch_size, num_nodes, 1, d_model].
        ...
        Encode the target data and compute the summary representation.
        - target_data (torch.Tensor): The target data with shape [batch_size, num_samples, num_nodes, 1].
        - summary_rep (torch.Tensor): The summary representation with shape [batch_size, num_nodes, 1, d_model].
        ...
    """

    def __init__(
        self,
        d_model,
        dim_feedforward,
        nhead,
        num_layers,
        num_nodes,
        device,
        dtype,
        emb_depth: int = 1,
        dropout: Optional[float] = 0.0,
    ):
        super(CausalInfEncoder, self).__init__()
        # Different MLP for context and target
        # Different embedding for everything.
        mlp = partial(build_mlp, dim_in=1, dim_hid=d_model, dim_out=d_model, depth=emb_depth)
        self.context_variable = mlp()
        self.context_intervention = mlp()
        self.context_outcome = mlp()
        self.target_variable = mlp()
        self.target_intervention = mlp()
        self.target_outcome = mlp()

        module = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=False,
            batch_first=True,
            norm_first=True,
            device=device,
            dtype=dtype,
        )
        encoderlayers = nn.ModuleList(
            [copy.deepcopy(module) for i in range(num_layers)]
        )
        self.encoder = CausalInfTransformerEncoder(
            encoder_layers=encoderlayers,
        )
        self.representation = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

    def embed(self, data, outcome_indices, intervention_indices, context=True):
        """
        Embed the target data into a d_model dimensional space.

        Has a different embedding for context and target data.

        Args:
        -----
        - data: torch.Tensor, shape [batch_size, num_samples, num_nodes, 1]
        - outcome_indices: torch.Tensor, shape [batch_size]
        - intervention_indices: torch.Tensor, shape [batch_size]

        Returns:
        --------
            embedding: torch.Tensor, shape [batch_size, num_samples, num_nodes, d_model]
        """
        batch_size, num_samples, num_nodes, _ = data.size()

        # Choose the correct MLPs based on whether the data is context or target.
        if context:
            var_emb = self.context_variable(data)
            int_emb = self.context_intervention(data)
            out_emb = self.context_outcome(data)
        else:
            var_emb = self.target_variable(data)
            int_emb = self.target_intervention(data)
            out_emb = self.target_outcome(data)

        # Replace embeddings at outcome and intervention indices.
        # Using vectorized indexing:
        batch_idx = torch.arange(batch_size, device=data.device)

        # Create masks for outcome and intervention indices.
        mask_outcome = torch.zeros_like(var_emb)
        mask_int = torch.zeros_like(var_emb)
        mask_outcome[batch_idx[:, None], :, outcome_indices[:, None], :] = 1.0
        mask_int[batch_idx[:, None], :, intervention_indices[:, None], :] = 1.0

        # Combine embeddings with a weighted sum.
        # Ensure that masks do not overlap or adjust weights appropriately if they do.
        # embedding = var_emb * (1 - mask_outcome - mask_int) + out_emb * mask_outcome + int_emb * mask_int
        embedding = var_emb
        return embedding

    def create_input(self, context_data, target_data):
        """
        Creates the full input by concatenating the context and target data.
        """
        full_data = torch.cat([context_data, target_data], dim=1)
        return full_data

    def create_sample_mask(self, context_data, target_data):
        """
        Create mask to ensure that
        - Context samples can attend to itself
        - Target samples only attent to the context samples
        """
        num_context = context_data.size(1)
        num_target = target_data.size(1)
        mask_size = num_context + num_target
        sample_mask = torch.zeros(
            mask_size, mask_size, device=context_data.device, dtype=context_data.dtype
        ).fill_(float("-inf"))
        # Context samples can attend to itself
        # target samples can only attend to context samples
        # The data is in the form [context, target]
        sample_mask[:, :num_context] = 0
        return sample_mask, num_target

    def encode(
        self,
        context_data: Tensor,
        target_data: Tensor,
        outcome_indices: Tensor,
        intervention_indices: Tensor,
        variable_mask: Optional[Tensor] = None,
    ):
        """
        Args:
        -----
        - context_data (torch.Tensor): The context data with shape
        [batch_size, num_context, num_nodes, 1].
        - target_data (torch.Tensor): The target data with shape
        [batch_size, num_target, num_nodes, 1]. This should be all zeros
        except for node that is meant to be the interventional variable.
        - outcome_indices (torch.Tensor, optional): The indices of the outcome
        with shape [batch_size]. Defaults to None.
        - variable_mask (torch.Tensor, optional): The mask that allows for
        different number of variables. Defaults to None.
        """
        context_embed = self.embed(
            data=context_data,
            outcome_indices=outcome_indices,
            intervention_indices=intervention_indices,
            context=True,
        )
        target_embed = self.embed(
            data=target_data,
            outcome_indices=outcome_indices,
            intervention_indices=intervention_indices,
            context=False,
        )
        # shape [batch_size, num_context + num_target, num_nodes, d_model]
        full_data = self.create_input(context_embed, target_embed)
        sample_mask, num_target = self.create_sample_mask(context_data, target_data)
        # shape [batch_size, num_samples, num_nodes, d_model]
        # Encode the data
        # shape [batch_size, num_samples, num_nodes, d_model]
        representation = self.encoder(
            full_data,
            sample_mask=sample_mask,
            variable_mask=variable_mask
        )

        # Compute the summary representation
        # shape [batch_size, num_target, num_nodes, d_model]
        summary_rep = representation[:, -num_target:, :, :]
        return summary_rep
