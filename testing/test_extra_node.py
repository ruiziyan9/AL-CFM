
import torch

from CITNP.models.extra_node_encoder import (
    InterventionSetEmbedding,
    ExtraNodeMixedCausalInfEncoder,
)
from CITNP.models.mixed_context_model import (
    ExtraNodeMixedContextMoGModel,
    MixedContextMoGModel,
    LayerwiseMixedContextMoGModel,
    load_upstream_checkpoint,
    resolve_model_class,
)
from CITNP.models.causalinferencemodel import (
    MoGCausalInferenceModel,
)
from CITNP.utils.configs import CausalInfModelConfig


def tiny_config():
    return CausalInfModelConfig(
        d_model=32,
        emb_depth=2,
        dim_feedforward=64,
        nhead=4,
        dropout=0.0,
        num_layers_encoder=4,
        num_nodes=4,
        sample_attn_mode="MHCA",
        linear_attention=False,
        mean_loss_across_samples=False,
        device="cpu",
        dtype="float32",
        num_mixture_components=3,
    )


def make_inputs():
    B, Nc, Nq, D = 2, 8, 3, 4

    context = torch.randn(B, Nc, D, 1)
    target = torch.randn(B, Nq, D, 1)

    intervention = torch.tensor([0, 3])
    outcome = torch.tensor([2, 1])

    mask = torch.zeros(B, Nc, D, dtype=torch.bool)

    # Two simultaneous interventions in context row 0.
    mask[0, 0, [0, 2]] = True

    # Single intervention in context row 1.
    mask[0, 1, 3] = True

    # Other context rows are observational.
    return context, target, intervention, outcome, mask


def test_registry():
    cls, name = resolve_model_class("extra_node")

    assert cls is ExtraNodeMixedContextMoGModel
    assert name == "extra_node"

    assert resolve_model_class("r1")[0] is MixedContextMoGModel
    assert (
        resolve_model_class("layerwise")[0]
        is LayerwiseMixedContextMoGModel
    )


def test_intervention_set_embedding():
    torch.manual_seed(0)

    embedding = InterventionSetEmbedding(
        num_nodes=4, d_model=32
    )

    mask = torch.zeros(1, 3, 4, dtype=torch.bool)

    # Row 0: observational
    # Row 1: do(X0, X2)
    # Row 2: do(X3)
    mask[0, 1, [0, 2]] = True
    mask[0, 2, 3] = True

    output = embedding(mask)

    assert output.shape == (1, 3, 32)
    assert torch.isfinite(output).all()

    # Shared identity, different context/query role.
    context = embedding(mask, is_query=False)
    query = embedding(mask, is_query=True)

    assert not torch.equal(context, query)


def test_extra_node_model_forward_backward():
    torch.manual_seed(0)

    model = ExtraNodeMixedContextMoGModel(tiny_config())
    model.train()

    assert isinstance(
        model.encoder,
        ExtraNodeMixedCausalInfEncoder,
    )

    context, target, intervention, outcome, mask = make_inputs()

    output = model(
        context_data=context,
        target_data=target,
        intervention_index=intervention,
        outcome_index=outcome,
        context_int_mask=mask,
    )

    assert torch.isfinite(output.pred_mean).all()
    assert torch.isfinite(output.pred_std).all()
    assert torch.isfinite(output.weights).all()

    loss = output.pred_mean.square().mean()
    loss.backward()

    grads = [
        p.grad for p in
        model.encoder.intervention_set.parameters()
        if p.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_upstream_initialization():
    upstream = MoGCausalInferenceModel(tiny_config())
    extra = ExtraNodeMixedContextMoGModel(tiny_config())

    result = load_upstream_checkpoint(
        extra, upstream.state_dict()
    )

    assert not result.unexpected_keys
    assert all(
        key.startswith("encoder.intervention_set.")
        for key in result.missing_keys
    )


def test_checkpoint_roundtrip():
    model = ExtraNodeMixedContextMoGModel(tiny_config())
    restored = ExtraNodeMixedContextMoGModel(tiny_config())

    restored.load_state_dict(model.state_dict(), strict=True)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[key], value
        )


def test_real_node_representation_shape():
    model = ExtraNodeMixedContextMoGModel(tiny_config())
    context, target, intervention, outcome, mask = make_inputs()

    # The encoder returns only real-variable representations.
    masked_target = model._mask_target_data(
        target_data=target,
        intervention_index=intervention,
    )

    representation, num_target = model.encoder.encode(
        context_data=context,
        target_data=masked_target,
        intervention_indices=intervention,
        outcome_indices=outcome,
        context_int_mask=mask,
    )

    assert representation.shape == (2, 11, 4, 32)
    assert num_target == 3
