"""Extra-node intervention encoding (intervention_encoding: extra_node).

The six role MLPs of upstream MACE-TNP encode the query roles exactly as in
R1. Instead of R1's per-cell flag, every row gets one extra node position:

    context cell     h_ij      = f_C,r_j(x_ij) + e_j
    context metadata h_i,META  = e_INT + sum_j m_ij e_j
    query cell       h_qj      = f_Q,r_j(x~_qj) + e_j
    query metadata   h_q,META  = f_Q,variable(0)       (no index embedding)

e_j are learned variable-index embeddings shared by real-variable and
metadata tokens; e_INT is a learned metadata-type embedding; m_i is the
row's context_int_mask. The transformer runs on D + 1 node positions; the
metadata position is dropped before returning, so the prediction head and
every downstream interface see D variables as before.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from CITNP.models.causaltransformercomponents import CausalInfEncoder
from CITNP.models.scale_conditioning import ScaleConditioning, check_scale_conditioning

INDEX_EMBEDDING_INIT_STD = 0.02


class ExtraNodeMixedCausalInfEncoder(CausalInfEncoder):
    """CausalInfEncoder with index embeddings and an intervention-metadata
    node. Has no R1 (`context_intervened`) or layer-wise parameters."""

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
        scale_conditioning: str = "none",
    ):
        super().__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers,
            num_nodes=num_nodes,
            device=device,
            dtype=dtype,
            emb_depth=emb_depth,
            dropout=dropout,
            sample_attn_mode=sample_attn_mode,
            linear_attention=linear_attention,
        )
        self.num_nodes = num_nodes
        self.index_embedding = nn.Embedding(num_nodes, d_model)
        self.int_meta_embedding = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.index_embedding.weight, std=INDEX_EMBEDDING_INIT_STD)
        nn.init.normal_(self.int_meta_embedding, std=INDEX_EMBEDDING_INIT_STD)
        # Created last: the other parameters' init does not depend on it.
        self.scale_conditioning = check_scale_conditioning(scale_conditioning)
        if self.scale_conditioning == "stats":
            self.scale_embedding = ScaleConditioning(d_model, emb_depth)

    def index_embeddings(self, num_nodes: int) -> Tensor:
        """E[:num_nodes], shape [num_nodes, d_model]."""
        if num_nodes > self.num_nodes:
            raise ValueError(
                f"Data has {num_nodes} variables but the model was built for "
                f"at most {self.num_nodes}"
            )
        return self.index_embedding.weight[:num_nodes]

    def context_metadata(
        self,
        context_int_mask: Optional[Tensor],
        batch_size: int,
        num_context: int,
        num_nodes: int,
    ) -> Tensor:
        """e_INT + m_i E for every context row -> [B, Nc, 1, d_model].
        None means every row is observational (as in R1)."""
        E = self.index_embeddings(num_nodes)
        meta = self.int_meta_embedding.expand(batch_size, num_context, -1)
        if context_int_mask is not None:
            meta = meta + context_int_mask.to(E.dtype) @ E
        return meta.unsqueeze(2)

    def query_metadata(self, query_data: Tensor) -> Tensor:
        """target_variable(0) for every query row -> [B, Nq, 1, d_model]."""
        zeros = query_data.new_zeros(*query_data.shape[:2], 1, 1)
        return self.target_variable(zeros)

    def _embed_rows(
        self,
        data: Tensor,
        outcome_indices: Tensor,
        intervention_indices: Tensor,
        context: bool,
        context_int_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Role embedding + index embedding, plus the metadata column:
        [B, N, D, 1] -> [B, N, D + 1, d_model]."""
        batch_size, num_samples, num_nodes, _ = data.shape
        emb = self.embed(
            data=data,
            outcome_indices=outcome_indices,
            intervention_indices=intervention_indices,
            context=context,
        ) + self.index_embeddings(num_nodes)
        if context:
            meta = self.context_metadata(
                context_int_mask, batch_size, num_samples, num_nodes
            )
        else:
            meta = self.query_metadata(data)
        return torch.cat([emb, meta], dim=2)

    def encode(
        self,
        context_data: Tensor,
        target_data: Tensor,
        outcome_indices: Tensor,
        intervention_indices: Tensor,
        target_train_data: Optional[Tensor] = None,
        variable_mask: Optional[Tensor] = None,
        context_int_mask: Optional[Tensor] = None,
        context_scale: Optional[Tensor] = None,
    ):
        """Same signature and return as MixedCausalInfEncoder.encode;
        representation is [B, Nc + Nq, D, d_model] (metadata dropped).

        variable_mask is passed through but unused by the upstream stack;
        if node masking is ever enabled, the metadata column must be valid.
        """
        num_nodes = context_data.shape[2]
        roles = dict(
            outcome_indices=outcome_indices,
            intervention_indices=intervention_indices,
        )
        context_embed = self._embed_rows(
            context_data, context=True, context_int_mask=context_int_mask, **roles
        )
        target_embed = self._embed_rows(target_data, context=False, **roles)
        target_train_embed = (
            None
            if target_train_data is None
            else self._embed_rows(target_train_data, context=False, **roles)
        )
        full_data = self.create_input(
            context_embed, target_embed, target_train_embed
        )
        if self.scale_conditioning == "stats":
            if context_scale is None:
                raise ValueError("scale_conditioning=stats needs context_scale")
            # the D variable columns only; the INT_META column gets nothing
            full_data = self.scale_embedding.add_to(full_data, context_scale)
        sample_mask, num_target = self.create_sample_mask(
            context_embed, target_embed, target_train_embed
        )
        representation = self.encoder(
            full_data,
            sample_mask=sample_mask,
            num_target=num_target,
            variable_mask=variable_mask,
        )
        return representation[:, :, :num_nodes], num_target
