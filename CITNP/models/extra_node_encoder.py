
"""Query-aware extra-node intervention encoding for mixed-context Causal PFNs.

One metadata token is appended along the node dimension of every
context/query sample.

Context metadata: set of nodes intervened on in that sample.
Query metadata: requested intervention target of the query.

The metadata token participates in attention but is removed before
the causal prediction head.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from CITNP.models.causaltransformercomponents import CausalInfEncoder


class InterventionSetEmbedding(nn.Module):
    """Map an intervention-set mask to a single embedding.

    mask: [B, N, D] bool
    output: [B, N, H]

    An all-False mask represents observational data.
    """

    def __init__(self, num_nodes: int, d_model: int):
        super().__init__()

        self.num_nodes = num_nodes

        # One learned identity embedding per possible SCM node.
        self.node_identity = nn.Embedding(num_nodes, d_model)

        # Shared target-set projection for context and query.
        self.projection = nn.Linear(
            d_model, d_model, bias=False
        )

        # 0: context metadata, 1: query metadata.
        self.role_embedding = nn.Embedding(2, d_model)

    def forward(
        self,
        mask: Tensor,
        is_query: bool = False,
    ) -> Tensor:
        if mask.ndim != 3:
            raise ValueError(
                "mask must have shape [B, N, D]"
            )

        B, N, D = mask.shape

        if D != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, got {D}"
            )

        if mask.dtype != torch.bool:
            raise TypeError("Intervention mask must be bool")

        # [D, H]
        identities = self.node_identity.weight

        # [B, N, D] @ [D, H] -> [B, N, H]
        aggregate = (
            mask.to(dtype=identities.dtype) @ identities
        )

        target_embedding = self.projection(aggregate)

        role_id = 1 if is_query else 0

        # [H] broadcasts across B and N.
        role = self.role_embedding.weight[role_id]

        return target_embedding + role


class ExtraNodeMixedCausalInfEncoder(CausalInfEncoder):
    """CausalInfEncoder with query-aware intervention metadata."""

    def __init__(
        self,
        d_model,
        dim_feedforward,
        nhead,
        num_layers,
        num_nodes,
        device,
        dtype,
        emb_depth=1,
        dropout=0.0,
        sample_attn_mode="MHSA",
        linear_attention=False,
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

        self.intervention_set = InterventionSetEmbedding(
            num_nodes=num_nodes,
            d_model=d_model,
        )

    @staticmethod
    def _append_token(
        embeddings: Tensor,
        metadata: Tensor,
    ) -> Tensor:
        # embeddings: [B, N, D, H]
        # metadata:   [B, N, H]
        return torch.cat(
            [embeddings, metadata.unsqueeze(2)],
            dim=2,
        )

    @staticmethod
    def _query_mask(
        intervention_indices: Tensor,
        num_samples: int,
        num_nodes: int,
        device: torch.device,
    ) -> Tensor:
        """Build a query intervention mask from existing [B] indices."""

        if intervention_indices.ndim != 1:
            raise ValueError(
                "Expected intervention_indices of shape [B]"
            )

        ids = intervention_indices.to(
            device=device, dtype=torch.long
        )
        if torch.any((ids < 0) | (ids >= num_nodes)):
            raise ValueError("Query intervention index out of range")

        B = ids.shape[0]

        mask = torch.zeros(
            B, num_samples, num_nodes,
            device=device,
            dtype=torch.bool,
        )

        mask.scatter_(
            dim=2,
            index=ids[:, None, None].expand(B, num_samples, 1),
            value=True,
        )

        return mask

    def encode(
        self,
        context_data: Tensor,
        target_data: Tensor,
        outcome_indices: Tensor,
        intervention_indices: Tensor,
        target_train_data: Optional[Tensor] = None,
        variable_mask: Optional[Tensor] = None,
        context_int_mask: Optional[Tensor] = None,
    ):
        B, Nc, D, _ = context_data.shape

        # 1. Preserve the existing value/role embeddings.
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

        # 2. Encode the observed intervention sets.
        if context_int_mask is None:
            context_int_mask = torch.zeros(
                B, Nc, D,
                device=context_data.device,
                dtype=torch.bool,
            )

        if context_int_mask.shape != (B, Nc, D):
            raise ValueError(
                "context_int_mask must have shape "
                f"{(B, Nc, D)}"
            )

        context_metadata = self.intervention_set(
            context_int_mask,
            is_query=False,
        )

        context_embed = self._append_token(
            context_embed,
            context_metadata,
        )

        # 3. Encode the requested query intervention.
        Nq = target_embed.shape[1]

        query_mask = self._query_mask(
            intervention_indices,
            Nq,
            D,
            target_embed.device,
        )

        query_metadata = self.intervention_set(
            query_mask,
            is_query=True,
        )

        target_embed = self._append_token(
            target_embed,
            query_metadata,
        )

        # 4. Optional training-target representation.
        if target_train_data is not None:
            target_train_embed = self.embed(
                data=target_train_data,
                outcome_indices=outcome_indices,
                intervention_indices=intervention_indices,
                context=False,
            )

            Nt = target_train_embed.shape[1]

            train_query_mask = self._query_mask(
                intervention_indices,
                Nt,
                D,
                target_train_embed.device,
            )

            train_metadata = self.intervention_set(
                train_query_mask,
                is_query=True,
            )

            target_train_embed = self._append_token(
                target_train_embed,
                train_metadata,
            )
        else:
            target_train_embed = None

        # 5. Reuse existing sample construction and attention.
        full_data = self.create_input(
            context_embed,
            target_embed,
            target_train_embed,
        )

        sample_mask, num_target = self.create_sample_mask(
            context_embed,
            target_embed,
            target_train_embed,
        )

        representation = self.encoder(
            full_data,
            sample_mask=sample_mask,
            num_target=num_target,
            variable_mask=variable_mask,
        )

        # 6. Discard the metadata token, keep all real nodes.
        representation = representation[:, :, :D, :]

        return representation, num_target
