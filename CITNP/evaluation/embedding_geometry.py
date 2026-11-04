"""Geometry of the extra-node intervention-metadata embeddings. Reads
parameters only; never runs the model.

In the extra-node encoding (CITNP/models/extra_node_encoder.py) the metadata
column of context row i is

    observational row             e_INT
    row intervened on set S       e_INT + sum_{j in S} e_j

with e_INT = `encoder.int_meta_embedding` and e_j = row j of
`encoder.index_embedding` (shared with the data cells of variable j). The
question is whether adding e_INT makes intervened rows hard to tell apart
from observational ones. The first encoder block is pre-norm, so attention
sees LayerNorm(token), not the token: LayerNorm removes the token's mean and
scale, so if e_INT dominates e_j then LayerNorm(e_INT + e_j) is close to
LayerNorm(e_INT). `metadata_embedding_geometry` reports norms and cosines of
the raw embeddings and, after the first block's input LayerNorm (affine
weights included, as the block applies it), the cosine of every intervened
token to the observational one and the distances between intervened tokens.
"""

import itertools
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from CITNP.models.extra_node_encoder import ExtraNodeMixedCausalInfEncoder


def first_block_input_normalisation(encoder: ExtraNodeMixedCausalInfEncoder):
    """(name, function [..., d_model] -> [..., d_model]) of what the first
    encoder block applies to its input tokens before attention.

    The first block is the first sample-attention layer of the stack (MHSA:
    `sample_layers[0]`; MHCA: the context self-attention `sample_layers[0]`).
    Pre-norm (norm_first=True, the default): its `norm1` LayerNorm, computed
    here in float64 with the layer's eps and affine weights. Post-norm: the
    identity (attention sees the raw token).
    """
    layer = encoder.encoder.sample_layers[0]
    if not layer.norm_first:
        return "none (post-norm block)", lambda tokens: tokens
    norm = layer.norm1
    weight = None if norm.weight is None else norm.weight.detach().double()
    bias = None if norm.bias is None else norm.bias.detach().double()

    def apply(tokens: Tensor) -> Tensor:
        return F.layer_norm(
            tokens, norm.normalized_shape, weight, bias, norm.eps
        )

    return "first block input LayerNorm (pre-norm block)", apply


def _cosine(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity along the last axis (broadcasting)."""
    return F.cosine_similarity(a, b, dim=-1, eps=1e-12)


def _min_pairwise_distance(tokens: Tensor) -> Optional[float]:
    """Smallest Euclidean distance between two rows of tokens [n, d]; None
    for fewer than two rows."""
    if tokens.shape[0] < 2:
        return None
    distances = torch.cdist(tokens[None], tokens[None])[0]
    upper = torch.triu_indices(len(tokens), len(tokens), offset=1)
    return float(distances[upper[0], upper[1]].min())


def _target_set_statistics(
    target_sets: list[tuple[int, ...]],
    int_meta: Tensor,
    index_embeddings: Tensor,
    normalise,
) -> dict:
    """Statistics of the metadata tokens e_INT + sum_{j in S} e_j for every
    S in target_sets (see `metadata_embedding_geometry`)."""
    if not target_sets:
        return {
            "target_sets": [],
            "raw_token_norms": [],
            "summed_index_embedding_norms": [],
            "projection_on_summed_index_embedding": [],
            "normalised_cosine_to_observational": [],
            "normalised_distance_to_observational": [],
            "min_pairwise_normalised_distance": None,
        }
    summed = torch.stack(
        [index_embeddings[list(S)].sum(0) for S in target_sets]
    )  # [n_sets, d]
    tokens = int_meta + summed
    normalised = normalise(tokens)
    normalised_observational = normalise(int_meta)
    summed_norms = summed.norm(dim=-1)
    projection = (tokens * summed).sum(-1) / summed_norms.clamp_min(1e-12)
    return {
        "target_sets": [list(S) for S in target_sets],
        "raw_token_norms": tokens.norm(dim=-1).tolist(),
        "summed_index_embedding_norms": summed_norms.tolist(),
        "projection_on_summed_index_embedding": projection.tolist(),
        "normalised_cosine_to_observational": _cosine(
            normalised, normalised_observational
        ).tolist(),
        "normalised_distance_to_observational": (
            normalised - normalised_observational
        )
        .norm(dim=-1)
        .tolist(),
        "min_pairwise_normalised_distance": _min_pairwise_distance(normalised),
    }


@torch.no_grad()
def metadata_embedding_geometry(
    encoder: ExtraNodeMixedCausalInfEncoder,
    num_variables: Optional[int] = None,
) -> dict:
    """Geometry of the metadata-column tokens of an extra-node encoder.

    Args:
    ----------
    encoder : ExtraNodeMixedCausalInfEncoder
        `model.encoder` of an extra-node model (trained or not).

    num_variables : int | None
        Number of variables D (default: the encoder's maximum, num_nodes).

    Returns:
    ----------
    A JSON-serialisable dict (plain floats, ints, strings and lists):
        num_variables, d_model : int
        normalisation : str
            What the first block applies to its input (see
            `first_block_input_normalisation`); "normalised" below means
            after it.
        int_meta_embedding_norm : float
            ||e_INT||.
        index_embedding_norms : list[float] [D]
            ||e_j||.
        cosine_int_meta_vs_index : list[float] [D]
            cos(e_INT, e_j) of the raw embeddings.
        observational_token_norm : float
            ||e_INT||, the raw observational metadata token.
        single_target, two_targets : dict
            One entry per target set S: all S = {j} (D sets), and all
            S = {j, k}, j < k (D (D - 1) / 2 sets; empty for D < 2). Lists
            are in the order of `target_sets`:
            target_sets                         list of [j] or [j, k]
            raw_token_norms                     ||e_INT + e_S||
            summed_index_embedding_norms        ||e_S||, e_S = sum_{j in S} e_j
            projection_on_summed_index_embedding
                                                <e_INT + e_S, e_S> / ||e_S||
            normalised_cosine_to_observational  cos(N(e_INT + e_S), N(e_INT))
            normalised_distance_to_observational
                                                ||N(e_INT + e_S) - N(e_INT)||
            min_pairwise_normalised_distance    min over S != S' of
                                                ||N(e_INT + e_S) -
                                                N(e_INT + e_S')|| (None for
                                                fewer than two sets)
            with N the first block's input normalisation.
    """
    if not isinstance(encoder, ExtraNodeMixedCausalInfEncoder):
        raise TypeError(
            "metadata_embedding_geometry needs an "
            "ExtraNodeMixedCausalInfEncoder (model.encoder of an extra_node "
            f"model), got {type(encoder).__name__}"
        )
    D = encoder.num_nodes if num_variables is None else int(num_variables)
    index_embeddings = encoder.index_embeddings(D).detach().double()  # [D, d]
    int_meta = encoder.int_meta_embedding.detach().double()  # [d]
    normalisation, normalise = first_block_input_normalisation(encoder)

    def statistics(size: int) -> dict:
        sets = list(itertools.combinations(range(D), size))
        return _target_set_statistics(
            sets, int_meta, index_embeddings, normalise
        )

    return {
        "num_variables": D,
        "d_model": int(int_meta.shape[0]),
        "normalisation": normalisation,
        "int_meta_embedding_norm": float(int_meta.norm()),
        "index_embedding_norms": index_embeddings.norm(dim=-1).tolist(),
        "cosine_int_meta_vs_index": _cosine(
            int_meta[None], index_embeddings
        ).tolist(),
        "observational_token_norm": float(int_meta.norm()),
        "single_target": statistics(1),
        "two_targets": statistics(2),
    }
