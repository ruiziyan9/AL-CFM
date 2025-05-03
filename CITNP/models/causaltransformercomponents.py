from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from CITNP.networks.attention_layers import (
    MultiHeadSelfAttentionLayer,
    MultiHeadCrossAttentionLayer,
)
from CITNP.models.nncomponents import build_mlp
from CITNP.utils.utils import (
    _get_clones,
    reshape_back_node_attention,
    reshape_back_sample_attention,
    reshape_for_node_attention,
    reshape_for_sample_attention,
)


class CausalInfTransformerEncoder(nn.Module):
    """
    Causal Inference Transformer that alternates attention between samples and nodes.
    """

    def __init__(
        self,
        sample_layers: nn.ModuleList,
        node_layers: nn.ModuleList,
        sample_attn_mode: str = "MHSA",
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ) -> None:
        super(CausalInfTransformerEncoder, self).__init__()
        assert len(sample_layers) + len(node_layers) > 0, (
            "Encoder must have at least one layer."
        )
        if sample_attn_mode == "MHSA":
            assert len(sample_layers) == len(node_layers), (
                "For MHSA, the number of sample and node layers must be equal.",
            )
            assert (len(sample_layers) + len(node_layers)) % 2 == 0, (
                "Encoder must have an even number of layers."
            )
        elif sample_attn_mode == "MHCA":
            assert len(sample_layers) == 2 * len(node_layers), (
                "For MHCA, the number of sample layers must be twice the number of node layers.",
            )
        self.sample_layers = sample_layers
        self.node_layers = node_layers
        self.sample_attn_mode = sample_attn_mode

        # Depending on attention mode, define how we iterate over the layers
        if self.sample_attn_mode == "MHSA":
            self.iterator = list(zip(self.sample_layers, self.node_layers))
        elif self.sample_attn_mode == "MHCA":
            # Flattened list of modules: [mhsa1, mhca1, mhsa2, mhca2, ...]
            # Group sample_layers as pairs: [(mhsa1, mhca1), (mhsa2, mhca2), ...]
            grouped = list(zip(self.sample_layers[::2], self.sample_layers[1::2]))
            self.iterator = list(zip(grouped, self.node_layers))
        else:
            raise ValueError(f"Unknown sample_attn_mode: {self.sample_attn_mode}")

    def forward(
        self,
        src: Tensor,
        sample_mask: Tensor,
        num_target: int,
        variable_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        # src: [batch_size, num_samples, num_nodes, d_model]
        # We need to reshape the tensor to [batch_size * num_nodes, num_samples, d_model]
        # to carry out attention over samples
        batch_size, num_samples, num_nodes, d_model = src.size()

        for sample_layer, node_layer in self.iterator:
            # Attention across samples
            # shape [batch_size * num_nodes, num_samples, d_model]
            src = reshape_for_sample_attention(
                src, batch_size, num_samples, num_nodes, d_model
            )
            if self.sample_attn_mode == "MHSA":
                src = sample_layer(
                    src,
                    mask=sample_mask,
                )
            elif self.sample_attn_mode == "MHCA":
                ctxt_src = src[:, :-num_target, :]
                tgt_src = src[:, -num_target:, :]
                # MHSA amongst context
                ctxt_src = sample_layer[0](ctxt_src)
                # MHCA between context and target
                tgt_src = sample_layer[1](tgt_src, ctxt_src)
                # Reshape the tensor back to [batch_size, num_samples, num_nodes, d_model]
                src = torch.cat([ctxt_src, tgt_src], dim=1)

            # Reshape the tensor back to [batch_size, num_samples, num_nodes, d_model]
            src = reshape_back_sample_attention(
                src, batch_size, num_samples, num_nodes, d_model
            )

            # Attention across nodes
            # shape [batch_size * num_samples, num_nodes, d_model]
            src = reshape_for_node_attention(
                src, batch_size, num_samples, num_nodes, d_model
            )
            # TODO: Add the node_src_key_padding_mask
            # Extra zeros for the query
            # node_src_key_padding_mask = variable_mask
            # if node_src_key_padding_mask is not None:
            #     node_src_key_padding_mask = node_src_key_padding_mask.reshape(
            #         batch_size * num_samples, num_nodes
            #     )
            src = node_layer(
                src,
                mask=None,
            )
            # Make masking position back to zero
            # if node_src_key_padding_mask is not None:
            #     bool_pad = node_src_key_padding_mask == -float("inf")
            #     src = src.masked_fill_(bool_pad.unsqueeze(-1), 0)
            # Reshape the tensor back to [batch_size, num_samples, num_nodes, d_model]
            src = reshape_back_node_attention(
                src, batch_size, num_samples, num_nodes, d_model
            )
        return src


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Causal Transformer for Decoders. There is no memory in the decoder.
    This will simply perform self-attention and feedforward operations.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(CausalTransformerDecoderLayer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            bias=bias,
            dtype=dtype,
        )
        self.dim_feedforward = dim_feedforward

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""
        Pass the inputs (and mask) through the decoder layer.

        It takes in memory but does nothing with it. This is to ensure
        compatibility with the nn.TransformerDecoder class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        assert memory is None, "Memory is not used in the decoder."

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm3(x + self._ff_block(x))
        return x


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
        sample_attn_mode: str = "MHSA",
        linear_attention: bool = False,
    ):
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        super(CausalInfEncoder, self).__init__()
        # Different MLP for context and target
        # Different embedding for everything.
        mlp = partial(
            build_mlp,
            dim_in=1,
            dim_hid=d_model,
            dim_out=d_model,
            depth=emb_depth,
        )
        self.context_variable = mlp()
        self.context_intervention = mlp()
        self.context_outcome = mlp()
        self.target_variable = mlp()
        self.target_intervention = mlp()
        self.target_outcome = mlp()

        module_mhsa = partial(
            MultiHeadSelfAttentionLayer,
            embed_dim=d_model,
            num_heads=nhead,
            head_dim=d_model // nhead,
            feedforward_dim=dim_feedforward,
            p_dropout=dropout,
            norm_first=True,
            activation=nn.GELU(),
            linear=linear_attention,
        )
        if sample_attn_mode == "MHCA":
            module_mhca = partial(
                MultiHeadCrossAttentionLayer,
                embed_dim=d_model,
                num_heads=nhead,
                head_dim=d_model // nhead,
                feedforward_dim=dim_feedforward,
                p_dropout=dropout,
                norm_first=True,
                activation=nn.GELU(),
                linear=linear_attention,
            )
            sample_layers = _get_clones(
                (module_mhsa, module_mhca), num_layers // 2
            )
        elif sample_attn_mode == "MHSA":
            sample_layers = _get_clones(module_mhsa, num_layers // 2)
        else:
            raise ValueError(
                f"Unknown sample attention mode: {sample_attn_mode}. "
                "Use either 'MHSA' or 'MHCA'."
            )

        node_layers = _get_clones(module_mhsa, num_layers // 2)
        self.encoder = CausalInfTransformerEncoder(
            sample_layers=sample_layers,
            node_layers=node_layers,
            sample_attn_mode=sample_attn_mode,
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

        # # Replace embeddings at outcome and intervention indices.
        # Using vectorized indexing:
        batch_idx = torch.arange(batch_size, device=data.device)

        # Create masks for outcome and intervention indices.
        mask_outcome = torch.zeros_like(var_emb)
        mask_int = torch.zeros_like(var_emb)
        mask_outcome[batch_idx[:, None], :, outcome_indices[:, None], :] = 1.0
        mask_int[batch_idx[:, None], :, intervention_indices[:, None], :] = 1.0
        # Combine embeddings with a weighted sum.
        # Ensure that masks do not overlap or adjust weights appropriately if they do.
        embedding = (
            var_emb * (1 - mask_outcome - mask_int)
            + out_emb * mask_outcome
            + int_emb * mask_int
        )

        return embedding

    def create_input(self, context_data, target_data, target_train_data):
        """
        Creates the full input by concatenating the context and target data.
        """
        if self.training and target_train_data is not None:
            full_data = torch.cat(
                [context_data, target_train_data, target_data], dim=1
            )
        else:
            full_data = torch.cat([context_data, target_data], dim=1)
        return full_data

    def create_sample_mask(self, context_data, target_data, target_train_data):
        """
        Create mask to ensure that
        - Context samples can attend to itself
        - Target train data can only attend to context samples
        - Target samples only attend to the context samples
        """
        num_context = context_data.size(1)
        num_target = target_data.size(1)
        # If training then target_train_data is not None
        if target_train_data is not None:
            mask_size = num_context + num_target + (num_target)
        else:
            mask_size = num_context + (num_target)
        sample_mask = torch.zeros(
            mask_size,
            mask_size,
            device=context_data.device,
            dtype=context_data.dtype,
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
        target_train_data: Optional[Tensor] = None,
        variable_mask: Optional[Tensor] = None,
    ):
        """
        Args:
        -----
        - context_embed (torch.Tensor): The context data with shape
            [batch_size, num_context, num_nodes, d_model].
        - target_embed (torch.Tensor): The target data with shape
            [batch_size, num_target, num_nodes, d_model]. This should be all zeros
            except for node that is meant to be the interventional variable.
        - target_train_embed (torch.Tensor): The target data with shape
            [batch_size, num_target, num_nodes, d_model]. This should be the same
            as target_data but with the true values of the outcome variable.
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
        if target_train_data is not None:
            target_train_embed = self.embed(
                data=target_train_data,
                outcome_indices=outcome_indices,
                intervention_indices=intervention_indices,
                context=False,
            )
        else:
            target_train_embed = None

        # shape [batch_size, num_context + num_target, num_nodes, d_model]
        full_data = self.create_input(
            context_embed, target_embed, target_train_embed
        )
        sample_mask, num_target = self.create_sample_mask(
            context_embed, target_embed, target_train_embed
        )
        # shape [batch_size, num_samples, num_nodes, d_model]
        # Encode the data
        # shape [batch_size, num_samples, num_nodes, d_model]
        representation = self.encoder(
            full_data,
            sample_mask=sample_mask,
            num_target=num_target,
            variable_mask=variable_mask,
        )

        # Compute the summary representation
        # shape [batch_size, num_target, num_nodes, d_model]
        return representation, num_target
