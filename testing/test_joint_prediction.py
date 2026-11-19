"""Tests for the joint prediction head (CITNP/models/prediction_heads.py):
p(X_{-T} | do(X_T = t), context) as a mixture of multivariate Gaussians with
low-rank-plus-diagonal covariances, for every intervention encoding.

Models and batches have five variables unless a test says otherwise. Every
test that builds a joint model runs on all five joint classes (the reg_axis
family also with one node-attention layer in the head).
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import (
    MixedRegimeConfig,
    MixedRegimeTaskDataset,
    collate_tasks,
    generate_task,
)
from CITNP.evaluation.metrics import (
    correlation_error,
    kl_truth_joint_model,
    kl_truth_product_of_marginals,
    sample_truth_joint,
    truth_joint_logpdf,
)
from CITNP.evaluation.model_api import (
    JointPrediction,
    _call,
    _model_inputs,
    build_model,
    joint_log_density,
    joint_marginal_mixture,
    joint_mixture_mean_and_covariance,
    predict_joint_mixture,
    predict_mixture,
    sample_joint_rows,
)
from CITNP.evaluation.oracle import LinearSCM
from CITNP.experiments.mixed_train import (
    MODEL_KEYS,
    apply_model_config,
    build_trainer,
    parse_args,
)
from CITNP.models.mixed_context_model import (
    INTERVENTION_ENCODINGS,
    JOINT_INTERVENTION_ENCODINGS,
    JointExtraNodeMixedContextMoGModel,
    JointLayerwiseMixedContextMoGModel,
    JointMixedContextMoGModel,
    JointRegimeAxisFullMixedContextMoGModel,
    JointRegimeAxisMixedContextMoGModel,
    resolve_model_class,
)
from CITNP.models.nncomponents import JointMoGPredictor
from CITNP.models.prediction_heads import check_joint_head_settings
from CITNP.trainer.mixed_context_trainer import (
    MixedContextTrainer,
    prepare_mixed_batch,
)
from CITNP.utils.config_files import load_config_mapping
from CITNP.utils.loss import (
    LOG_SQRT_TWO_PI,
    JointMixtureGaussianLoss,
    joint_mixture_log_density,
    pad_intervened_variables,
)
from CITNP.utils.outputs import JointModelOutput
from CITNP.utils.utils import convert_trainerdict_to_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import tiny_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "CITNP" / "experiments" / "configs"
D = 5
RANK = 3
REG_AXIS_FAMILY = (
    JointRegimeAxisMixedContextMoGModel,
    JointRegimeAxisFullMixedContextMoGModel,
)
JOINT_CLASSES = (
    JointMixedContextMoGModel,
    JointLayerwiseMixedContextMoGModel,
    JointExtraNodeMixedContextMoGModel,
    *REG_AXIS_FAMILY,
)
# (joint class, joint_head_node_attention_layers)
CASES = [(cls, 0) for cls in JOINT_CLASSES] + [
    (cls, 1) for cls in REG_AXIS_FAMILY
]
CASE_IDS = [f"{cls.__name__}-attn{layers}" for cls, layers in CASES]
EQUIVARIANT_CASES = [
    case for case in CASES if case[0] is not JointExtraNodeMixedContextMoGModel
]
EQUIVARIANT_IDS = [
    f"{cls.__name__}-attn{layers}" for cls, layers in EQUIVARIANT_CASES
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def joint_config(layers: int = 0, num_nodes: int = D, rank: int = RANK):
    cfg = tiny_config(num_nodes)
    cfg.prediction_head = "joint"
    cfg.joint_covariance_rank = rank
    cfg.joint_head_node_attention_layers = layers
    return cfg


def build(cls, layers=0, seed=0, num_nodes=D, double=False):
    torch.manual_seed(seed)
    model = cls(joint_config(layers, num_nodes)).eval()
    return model.double() if double else model


def perturb(model, seed=3, size=0.1):
    """Move every parameter by seeded noise, so zero-initialised layers (flag,
    injection, scale embedding) act on the output."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(size * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def context_mask(B, Nc, num_nodes, seed=0):
    """Observational rows, two single-target regimes and one two-target
    regime per task (targets drawn per task)."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(B, Nc, num_nodes, dtype=torch.bool)
    block = Nc // 6
    for b in range(B):
        a, c, e = torch.randperm(num_nodes, generator=g)[:3].tolist()
        mask[b, 3 * block : 4 * block, a] = True
        mask[b, 4 * block : 5 * block, c] = True
        mask[b, 5 * block :, [a, e]] = True
    return mask


def joint_inputs(B=4, Nc=24, Nq=6, num_nodes=D, seed=1, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    scale = torch.stack(
        [0.3 * torch.randn(B, num_nodes, generator=g),
         torch.randn(B, num_nodes, generator=g)],
        dim=-1,
    )  # fmt: skip
    return {
        "context_data": torch.randn(B, Nc, num_nodes, 1, generator=g).to(dtype),
        "target_data": torch.randn(B, Nq, num_nodes, 1, generator=g).to(dtype),
        "intervention_index": torch.tensor([0, 2, 4, 1, 3, 0])[:B] % num_nodes,
        "context_int_mask": context_mask(B, Nc, num_nodes, seed),
        "context_scale": scale.to(dtype),
    }


def random_output(B=3, Nq=4, K=3, num_nodes=D, R=RANK, seed=0, mask=None):
    g = torch.Generator().manual_seed(seed)
    if mask is None:
        mask = torch.zeros(B, num_nodes, dtype=torch.bool)
        mask[torch.arange(B), torch.arange(B) % num_nodes] = True
    return JointModelOutput(
        component_mean=torch.randn(B, Nq, K, num_nodes, generator=g,
                                   dtype=torch.float64),
        covariance_factor=torch.randn(B, Nq, K, num_nodes, R, generator=g,
                                      dtype=torch.float64),
        covariance_diagonal=0.1 + torch.rand(B, Nq, K, num_nodes, generator=g,
                                             dtype=torch.float64),
        mixture_logits=torch.randn(B, Nq, K, generator=g, dtype=torch.float64),
        intervened_mask=mask,
    )  # fmt: skip


def brute_force_log_density(out: JointModelOutput, target: torch.Tensor):
    """Gather the non-intervened variables, evaluate every component with
    MultivariateNormal and combine with logsumexp: [B, Nq] (float64)."""
    B, Nq, K, _ = out.component_mean.shape
    result = torch.zeros(B, Nq, dtype=torch.float64)
    log_w = torch.log_softmax(out.mixture_logits.double(), dim=-1)
    for b in range(B):
        keep = torch.nonzero(~out.intervened_mask[b]).flatten()
        for q in range(Nq):
            parts = []
            for k in range(K):
                F = out.covariance_factor[b, q, k].double()[keep]
                cov = F @ F.T + torch.diag(
                    out.covariance_diagonal[b, q, k].double()[keep]
                )
                mvn = torch.distributions.MultivariateNormal(
                    out.component_mean[b, q, k].double()[keep], cov
                )
                parts.append(mvn.log_prob(target[b, q].double()[keep]))
            result[b, q] = torch.logsumexp(torch.stack(parts) + log_w[b, q], 0)
    return result


def loss_vector(out, target):
    """Per-element JointMixtureGaussianLoss with sum over query rows [B]."""
    loss = JointMixtureGaussianLoss(mean_loss_across_samples=False, reduce="sum")
    return loss.calculate_loss(out, target, test=True)


def joint_prediction_from(out: JointModelOutput) -> JointPrediction:
    return JointPrediction(
        component_mean=out.component_mean,
        covariance_factor=out.covariance_factor,
        covariance_diagonal=out.covariance_diagonal,
        log_weights=torch.log_softmax(out.mixture_logits, dim=-1),
        intervened_mask=out.intervened_mask,
    )


def dense_covariance(jp) -> torch.Tensor:
    F = jp.covariance_factor
    return F @ F.transpose(-1, -2) + torch.diag_embed(jp.covariance_diagonal)


def as_double(out: JointModelOutput) -> JointModelOutput:
    return JointModelOutput(
        component_mean=out.component_mean.double(),
        covariance_factor=out.covariance_factor.double(),
        covariance_diagonal=out.covariance_diagonal.double(),
        mixture_logits=out.mixture_logits.double(),
        intervened_mask=out.intervened_mask,
    )


# Small three-variable linear-Gaussian prior for the joint evaluation tests
# (normalise: false, as the closed-form truth needs).
SMALL_PRIOR: dict[str, Any] = dict(
    function_generator="linear_iw",
    alpha0=1.5,
    num_variables=[3],
    graph_type=["ER"],
    graph_degrees={3: [1, 2, 3]},
    normalise=False,
    k_max=2,
    n_ctx_range=(30, 40),
    n_qry=6,
    qry_mode="grouped",
    n_obs_min=10,
)


def grouped_cfg_d5(**kw) -> MixedRegimeConfig:
    base: dict[str, Any] = dict(
        function_generator="linearfixedstd",
        num_variables=[D],
        graph_type=["ER"],
        graph_degrees={D: [3, 5]},
        n_ctx_range=(16, 64),
        n_qry=60,
        qry_mode="grouped",
        qry_groups=[1, 2, 3, 4, 5],
    )
    base.update(kw)
    return MixedRegimeConfig(**base)


def grouped_batch_d5(G=5, B=2, n_ctx=40, seed0=0):
    cfg = grouped_cfg_d5()
    tasks = [
        generate_task(cfg, seed0 + s, num_variables=D, n_ctx=n_ctx, n_groups=G)
        for s in range(B)
    ]
    return collate_tasks(tasks)


# --------------------------------------------------------------------------- #
# 1. Head shapes and constraints
# --------------------------------------------------------------------------- #
def test_head_shapes_and_constraints():
    K, R, minimum = 3, 4, 1e-3
    head = JointMoGPredictor(
        d_model=16, depth=2, num_mixture_components=K, covariance_rank=R,
        minimum_variance=minimum, node_attention_layers=1, nhead=4,
        dim_feedforward=32,
    )  # fmt: skip
    for num_nodes in (5, 3):
        x = torch.randn(2, 7, num_nodes, 16)
        mean, factor, diagonal, logits = head(x)
        assert mean.shape == (2, 7, K, num_nodes)
        assert factor.shape == (2, 7, K, num_nodes, R)
        assert diagonal.shape == (2, 7, K, num_nodes)
        assert logits.shape == (2, 7, K)
        assert (diagonal >= minimum).all()
    # no parameter depends on the number of variables
    assert all(D not in p.shape for p in head.parameters())


def test_head_initial_diagonal_scale():
    head = JointMoGPredictor(
        d_model=16, depth=2, num_mixture_components=2, covariance_rank=2,
        minimum_variance=1e-4, node_attention_layers=0, nhead=4,
        dim_feedforward=32,
    )  # fmt: skip
    _, _, diagonal, _ = head(torch.randn(3, 4, D, 16))
    # about softplus(0) = 0.693 at initialisation, not the floor
    assert torch.allclose(diagonal, torch.full_like(diagonal, 0.693), atol=0.05)


# --------------------------------------------------------------------------- #
# 2-4. Loss: brute force, unused intervened outputs, padding
# --------------------------------------------------------------------------- #
def test_loss_matches_brute_force():
    out = random_output(seed=1)
    target = torch.randn(3, 4, D, dtype=torch.float64)
    expected = -brute_force_log_density(out, target).sum(1)
    torch.testing.assert_close(loss_vector(out, target), expected, atol=1e-5,
                               rtol=0)  # fmt: skip
    # reductions as MixtureGaussianLoss: mean over rows, then over B
    mean_loss = JointMixtureGaussianLoss(
        mean_loss_across_samples=True, reduce="mean"
    ).calculate_loss(out, target)
    torch.testing.assert_close(
        mean_loss, -brute_force_log_density(out, target).mean(1).mean(0),
        atol=1e-5, rtol=0,
    )  # fmt: skip


def test_intervened_outputs_do_not_matter():
    out = random_output(seed=2)
    target = torch.randn(3, 4, D, dtype=torch.float64)
    base = loss_vector(out, target)
    mask = out.intervened_mask[:, None, None, :]  # [B, 1, 1, D]
    changed = JointModelOutput(
        component_mean=torch.where(mask, 50.0, out.component_mean),
        covariance_factor=torch.where(mask[..., None], -7.0,
                                      out.covariance_factor),
        covariance_diagonal=torch.where(mask, 123.0, out.covariance_diagonal),
        mixture_logits=out.mixture_logits,
        intervened_mask=out.intervened_mask,
    )  # fmt: skip
    torch.testing.assert_close(loss_vector(changed, target), base, atol=1e-10,
                               rtol=0)  # fmt: skip
    other_target = torch.where(out.intervened_mask[:, None, :], 99.0, target)
    torch.testing.assert_close(loss_vector(out, other_target), base, atol=1e-10,
                               rtol=0)  # fmt: skip


def _padded_log_density_without_constant(out, target):
    mean, factor, diagonal = pad_intervened_variables(
        out.component_mean, out.covariance_factor, out.covariance_diagonal,
        out.intervened_mask, target,
    )  # fmt: skip
    cov = factor @ factor.transpose(-1, -2) + torch.diag_embed(diagonal)
    mixture = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=out.mixture_logits),
        torch.distributions.MultivariateNormal(mean, cov),
    )
    return mixture.log_prob(target)


@pytest.mark.parametrize(
    "mask_rows",
    [
        [[0], [2], [4]],  # one intervened variable per row
        [[0, 1], [2, 4], [1, 3]],  # two per row
        [[0], [1, 3], [4], [0, 2]],  # one and two mixed, different positions
    ],
    ids=["one", "two", "mixed"],
)
def test_padding_equals_gathered_density(mask_rows):
    B = len(mask_rows)
    mask = torch.zeros(B, D, dtype=torch.bool)
    for b, cols in enumerate(mask_rows):
        mask[b, cols] = True
    out = random_output(B=B, seed=5, mask=mask)
    target = torch.randn(B, 4, D, dtype=torch.float64)
    gathered = brute_force_log_density(out, target)
    padded = joint_mixture_log_density(out, target)  # one batched call
    torch.testing.assert_close(padded, gathered, atol=1e-8, rtol=0)
    # without the constant the difference is 0.5 log(2 pi) per dummy variable
    raw = _padded_log_density_without_constant(out, target)
    count = mask.sum(-1).double()[:, None]
    torch.testing.assert_close(
        gathered - raw, LOG_SQRT_TWO_PI * count.expand_as(raw), atol=1e-8,
        rtol=0,
    )  # fmt: skip


# --------------------------------------------------------------------------- #
# 5. Outcome-free embedding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls, layers", CASES, ids=CASE_IDS)
def test_outcome_free(cls, layers):
    model = perturb(build(cls, layers))
    inp = joint_inputs()
    calls = []
    hooks = [
        getattr(model.encoder, name).register_forward_hook(
            lambda *_, name=name: calls.append(name)
        )
        for name in ("context_outcome", "target_outcome")
    ]
    try:
        with torch.no_grad():
            reference = model(**inp, outcome_index=None)
            for Y in ([1, 3, 0, 2], [4, 4, 4, 4]):
                out = model(**inp, outcome_index=torch.tensor(Y))
                for name in ("component_mean", "covariance_factor",
                             "covariance_diagonal", "mixture_logits"):
                    assert torch.equal(getattr(out, name),
                                       getattr(reference, name)), name
    finally:
        for h in hooks:
            h.remove()
    assert calls == []
    assert isinstance(reference, JointModelOutput)
    expected_mask = torch.nn.functional.one_hot(inp["intervention_index"], D)
    assert torch.equal(reference.intervened_mask, expected_mask.bool())
    # the outcome networks are kept, so encoder parameter names match the
    # marginal model's
    assert hasattr(model.encoder, "context_outcome")


# --------------------------------------------------------------------------- #
# 6. Node-permutation equivariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls, layers", EQUIVARIANT_CASES, ids=EQUIVARIANT_IDS)
def test_node_permutation_equivariance(cls, layers):
    model = perturb(build(cls, layers, double=True))
    inp = joint_inputs(dtype=torch.float64)
    perm = torch.tensor([3, 0, 4, 2, 1])  # new position i holds old variable
    inverse = torch.argsort(perm)
    permuted = {
        "context_data": inp["context_data"][:, :, perm],
        "target_data": inp["target_data"][:, :, perm],
        "intervention_index": inverse[inp["intervention_index"]],
        "context_int_mask": inp["context_int_mask"][:, :, perm],
        "context_scale": inp["context_scale"][:, perm],
    }
    with torch.no_grad():
        a = model(**inp)
        b = model(**permuted)
    torch.testing.assert_close(b.component_mean, a.component_mean[..., perm],
                               atol=1e-5, rtol=0)  # fmt: skip
    torch.testing.assert_close(b.covariance_factor,
                               a.covariance_factor[..., perm, :],
                               atol=1e-5, rtol=0)  # fmt: skip
    torch.testing.assert_close(b.covariance_diagonal,
                               a.covariance_diagonal[..., perm],
                               atol=1e-5, rtol=0)  # fmt: skip
    torch.testing.assert_close(b.mixture_logits, a.mixture_logits, atol=1e-5,
                               rtol=0)  # fmt: skip


# --------------------------------------------------------------------------- #
# 7, 8, 18. Marginals, moments and sampling of a JointPrediction
# --------------------------------------------------------------------------- #
def test_marginal_consistency():
    out = random_output(seed=7)
    jp = joint_prediction_from(out)
    cov = dense_covariance(jp)
    for Y in range(D):
        valid = ~out.intervened_mask[:, Y]
        if not valid.all():
            with pytest.raises(ValueError, match="intervened"):
                joint_marginal_mixture(jp, Y)
        queries = torch.nonzero(valid).flatten()
        sub = jp.select(queries)
        mu, sd, logw = joint_marginal_mixture(sub, Y)
        torch.testing.assert_close(mu, sub.component_mean[..., Y])
        torch.testing.assert_close(sd**2, cov[queries][..., Y, Y])
        torch.testing.assert_close(logw, sub.log_weights)


def test_sampling_single_component():
    g = torch.Generator().manual_seed(0)
    Q, n_t, R = 2, 2, 3
    mask = torch.zeros(Q, D, dtype=torch.bool)
    mask[0, 1] = mask[1, 4] = True
    jp = JointPrediction(
        component_mean=torch.randn(Q, n_t, 1, D, generator=g, dtype=torch.float64),
        covariance_factor=torch.randn(Q, n_t, 1, D, R, generator=g,
                                      dtype=torch.float64),
        covariance_diagonal=0.2 + torch.rand(Q, n_t, 1, D, generator=g,
                                             dtype=torch.float64),
        log_weights=torch.zeros(Q, n_t, 1, dtype=torch.float64),
        intervened_mask=mask,
    )  # fmt: skip
    t = np.array([-1.5, 2.0])
    rows = sample_joint_rows(jp, t, 40000, torch.Generator().manual_seed(1))
    assert rows.shape == (Q, n_t, 40000, D)
    assert (rows[0, :, :, 1] == torch.tensor(t)[:, None]).all()
    assert (rows[1, :, :, 4] == torch.tensor(t)[:, None]).all()
    cov = dense_covariance(jp)[:, :, 0]
    for q in range(Q):
        keep = torch.nonzero(~mask[q]).flatten()
        for i in range(n_t):
            x = rows[q, i][:, keep]
            torch.testing.assert_close(
                x.mean(0), jp.component_mean[q, i, 0, keep], atol=0.08, rtol=0
            )
            torch.testing.assert_close(
                torch.cov(x.T), cov[q, i][keep][:, keep], atol=0.15, rtol=0.05
            )


def test_mixture_sampling_and_moments():
    Q, n_t, K = 1, 1, 2
    mask = torch.zeros(Q, D, dtype=torch.bool)
    mask[0, 2] = True
    means = torch.stack([torch.full((D,), -10.0), torch.full((D,), 10.0)])
    g = torch.Generator().manual_seed(4)
    jp = JointPrediction(
        component_mean=means.double()[None, None],
        covariance_factor=0.5 * torch.randn(Q, n_t, K, D, 2, generator=g,
                                            dtype=torch.float64),
        covariance_diagonal=torch.full((Q, n_t, K, D), 0.3, dtype=torch.float64),
        log_weights=torch.log(torch.tensor([[[0.3, 0.7]]], dtype=torch.float64)),
        intervened_mask=mask,
    )  # fmt: skip
    rows = sample_joint_rows(jp, np.array([0.5]), 50000,
                             torch.Generator().manual_seed(2))[0, 0]  # fmt: skip
    keep = torch.tensor([0, 1, 3, 4])
    assert (rows[:, 2] == 0.5).all()
    # one component per row: all coordinates of a row on the same side
    signs = torch.sign(rows[:, keep])
    assert (signs == signs[:, :1]).all()
    assert abs((signs[:, 0] > 0).double().mean().item() - 0.7) < 0.01
    mean, cov = joint_mixture_mean_and_covariance(jp)
    torch.testing.assert_close(rows[:, keep].mean(0), mean[0, 0, keep],
                               atol=0.2, rtol=0)  # fmt: skip
    torch.testing.assert_close(torch.cov(rows[:, keep].T),
                               cov[0, 0][keep][:, keep], atol=0.5, rtol=0.02)  # fmt: skip
    assert (cov[0, 0, 2] == 0).all() and (cov[0, 0, :, 2] == 0).all()


# --------------------------------------------------------------------------- #
# 9, 21. Registry
# --------------------------------------------------------------------------- #
def test_joint_registry_matches_marginal_registry():
    assert set(JOINT_INTERVENTION_ENCODINGS) == set(INTERVENTION_ENCODINGS)
    for name, cls in JOINT_INTERVENTION_ENCODINGS.items():
        assert issubclass(cls, INTERVENTION_ENCODINGS[name])
        assert cls.prediction_head == "joint"


@pytest.mark.parametrize("name", sorted(INTERVENTION_ENCODINGS))
def test_every_encoding_has_a_joint_class(name):
    cls, got = resolve_model_class(name, prediction_head="joint")
    assert cls is JOINT_INTERVENTION_ENCODINGS[name] and got == name


def test_joint_default_is_shared_layerwise():
    cls, name = resolve_model_class(prediction_head="joint")
    assert cls is JointLayerwiseMixedContextMoGModel and name == "layerwise"
    model = cls(joint_config())
    assert model.encoder.share_intervention_encoder is True


# (model_class, prediction_head, expected class name or ValueError)
REGISTRY_TABLE = [
    (None, None, "LayerwiseMixedContextMoGModel"),
    (None, "marginal", "LayerwiseMixedContextMoGModel"),
    ("MixedContextMoGModel", None, "MixedContextMoGModel"),
    ("MixedContextMoGModel", "marginal", "MixedContextMoGModel"),
    (None, "joint", "JointLayerwiseMixedContextMoGModel"),
    ("MixedContextMoGModel", "joint", "JointMixedContextMoGModel"),
    ("RegimeAxisMixedContextMoGModel", "joint",
     "JointRegimeAxisMixedContextMoGModel"),
    ("JointExtraNodeMixedContextMoGModel", None,
     "JointExtraNodeMixedContextMoGModel"),
    ("JointExtraNodeMixedContextMoGModel", "joint",
     "JointExtraNodeMixedContextMoGModel"),
    ("JointLayerwiseMixedContextMoGModel", "marginal", ValueError),
    ("MixedContextModel", "joint", ValueError),
    ("LayerwiseMixedContextModel", "joint", ValueError),
]  # fmt: skip


def _model_yaml(tmp_path, **extra) -> Path:
    c = tiny_config()
    settings = {
        "d_model": c.d_model,
        "emb_depth": c.emb_depth,
        "dim_feedforward": c.dim_feedforward,
        "nhead": c.nhead,
        "num_layers_encoder": c.num_layers_encoder,
        "sample_attn_mode": c.sample_attn_mode,
        "num_mixture_components": c.num_mixture_components,
        **{k: v for k, v in extra.items() if v is not None},
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def _train_args(model_config=None, **kw) -> argparse.Namespace:
    base: dict[str, Any] = {k: None for k in MODEL_KEYS}
    base["model_config"] = None if model_config is None else str(model_config)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("model_class, head, expected", REGISTRY_TABLE)
def test_registry_table_through_both_paths(tmp_path, model_class, head,
                                           expected):  # fmt: skip
    path = _model_yaml(tmp_path, model_class=model_class, prediction_head=head)
    args = _train_args(path)
    apply_model_config(args)
    if expected is ValueError:
        with pytest.raises(ValueError):
            resolve_model_class(
                args.intervention_encoding, args.model_class,
                args.prediction_head,
            )  # fmt: skip
        with pytest.raises(ValueError):
            build_model(str(path), "cpu", num_nodes=D)
        return
    train_cls, _ = resolve_model_class(
        args.intervention_encoding, args.model_class, args.prediction_head
    )
    eval_model = build_model(str(path), "cpu", num_nodes=D)
    assert train_cls.__name__ == expected
    assert type(eval_model) is train_cls


def test_joint_class_reports_marginal_encoding():
    cls, name = resolve_model_class(
        model_class="JointRegimeAxisFullMixedContextMoGModel"
    )
    assert cls is JointRegimeAxisFullMixedContextMoGModel
    assert name == "reg_axis_full"
    with pytest.raises(ValueError, match="contradicts"):
        resolve_model_class(
            "layerwise", "JointRegimeAxisFullMixedContextMoGModel"
        )


# --------------------------------------------------------------------------- #
# 10. Trainer
# --------------------------------------------------------------------------- #
def test_prepare_without_outcome():
    batch = grouped_batch_d5(G=5)
    with_y = prepare_mixed_batch(batch, torch.float32, False)
    without = prepare_mixed_batch(batch, torch.float32, False,
                                  draw_outcome=False)  # fmt: skip
    assert without["outcome_index"] is None
    assert with_y["outcome_index"] is not None
    assert with_y["meta"] is without["meta"]  # the batch's own list
    for key, value in with_y.items():
        if key in ("outcome_index", "meta", "norm"):
            continue
        if torch.is_tensor(value):
            assert torch.equal(value, without[key]), key
        else:
            assert value == without[key], key
    for a, b in zip(with_y["norm"], without["norm"]):
        assert torch.equal(a, b)


def _joint_trainer(model, tmp_path):
    cfg = grouped_cfg_d5(qry_groups=[5])
    datasets = [
        MixedRegimeTaskDataset(
            cfg, 2, base_seed=0, split=split, iterations_per_epoch=1,
            fresh_each_epoch=False,
        )
        for split in ("train", "val", "test")
    ]  # fmt: skip
    dataconfig, optimconfig, trainingconfig, loggingconfig = (
        convert_trainerdict_to_config(
            {
                "batch_size": 2, "num_workers": 0, "cntxt_split": [0.05, 0.75],
                "sample_size": 0, "train_dtype": "float32",
                "eval_dtype": "float32", "pin_memory": False, "normalise": True,
                "optimizer": torch.optim.Adam(model.parameters(), lr=1e-3),
                "learning_rate": 1e-3, "lr_warmup_ratio": 0.0, "epochs": 1,
                "gradient_clip_val": 1.0, "device": "cpu",
                "save_dir": Path(tmp_path), "use_wandb": False,
                "log_step": 1000, "save_checkpoint_every_n_steps": 1000,
                "plot_validation_samples": False, "num_validation_plots": 0,
            }
        )
    )  # fmt: skip
    return MixedContextTrainer(
        train_dataset=datasets[0],
        validation_dataset=datasets[1],
        test_dataset=datasets[2],
        model=model,
        dataconfig=dataconfig,
        optimizerconfig=optimconfig,
        trainingconfig=trainingconfig,
        loggingconfig=loggingconfig,
        callbacks=[],
        seed=0,
    )


@pytest.mark.parametrize("cls, layers", CASES, ids=CASE_IDS)
def test_trainer_step_and_test_epoch(cls, layers, tmp_path):
    model = build(cls, layers).train()
    trainer = _joint_trainer(model, tmp_path)
    prep = trainer._prepare_batch(grouped_batch_d5(G=5), dtype=torch.float32)
    assert prep["outcome_index"] is None and prep["G"] == 5
    logs = trainer.train_step(prep)
    assert logs["skipped"] == 0.0 and math.isfinite(logs["loss"])
    head_grads = [p.grad for p in model.predictor.parameters()]
    assert all(g is not None for g in head_grads)
    assert any(g.abs().sum() > 0 for g in head_grads)
    metrics = trainer.test()
    assert math.isfinite(metrics["test_loss_all_mean"])
    for key in ("downstream", "upstream", "independent"):
        assert metrics[f"test_loss_{key}_mean"] is None
    assert any(k.startswith("test_loss_G=") for k in metrics)


# --------------------------------------------------------------------------- #
# 11, 17. Evaluation interface on the raw scale
# --------------------------------------------------------------------------- #
def _raw_context(seed=0, n=60):
    rng = np.random.default_rng(seed)
    x = 40.0 + 7.0 * rng.standard_normal((n, D)) * np.arange(1, D + 1)
    m = np.zeros((n, D), dtype=bool)
    m[45:, 1] = True
    x[45:, 1] = rng.normal(40.0, 20.0, 15)
    return x, m


def _standardised_outputs(model, x, m, T, t):
    inp, mean, std = _model_inputs(x, m, T, None, t, zero_flags=False,
                                   normalise=True)  # fmt: skip
    with torch.no_grad():
        out = _call(model, inp, "cpu")
    return out, mean.double(), std.double()


@pytest.mark.parametrize("cls, layers", CASES, ids=CASE_IDS)
def test_predict_mixture_dispatch_by_value(cls, layers):
    model = perturb(build(cls, layers))
    x, m = _raw_context()
    t = np.array([20.0, 40.0, 60.0])
    jp = predict_joint_mixture(model, x, m, [0, 3], t, device="cpu")
    out, mean, std = _standardised_outputs(model, x, m, [0, 3], t)
    out = as_double(out)
    cov_std = dense_covariance(joint_prediction_from(out))
    T_list, Y_list = np.array([0, 0, 3, 3]), np.array([1, 4, 2, 0])
    mu, sd, logw = predict_mixture(model, x, m, T_list, Y_list, t, device="cpu")
    assert mu.shape == (4, len(t), 3) and mu.dtype == torch.float64
    for i, (T, Y) in enumerate(zip(T_list, Y_list)):
        q = 0 if T == 0 else 1
        expected_mu = out.component_mean[q, ..., Y].double() * std[Y] + mean[Y]
        expected_var = cov_std[q, ..., Y, Y] * std[Y] ** 2
        torch.testing.assert_close(mu[i], expected_mu, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(sd[i] ** 2, expected_var, atol=1e-6,
                                   rtol=1e-6)  # fmt: skip
        torch.testing.assert_close(logw[i], jp.log_weights[q])
    # two outcomes of one treatment read the same joint prediction
    torch.testing.assert_close(logw[0], logw[1], atol=0, rtol=0)
    with pytest.raises(ValueError, match="intervened"):
        predict_mixture(model, x, m, [3], [3], t, device="cpu")


@pytest.mark.parametrize("cls, layers", CASES, ids=CASE_IDS)
def test_raw_scale_parameters_and_density(cls, layers):
    model = perturb(build(cls, layers))
    x, m = _raw_context(seed=1)
    t = np.array([25.0, 55.0])
    T = [2, 4]
    jp = predict_joint_mixture(model, x, m, T, t, device="cpu")
    out, mean, std = _standardised_outputs(model, x, m, T, t)
    out = as_double(out)
    S = torch.diag(std)
    cov_std = dense_covariance(joint_prediction_from(out))
    torch.testing.assert_close(dense_covariance(jp), S @ cov_std @ S,
                               atol=1e-6, rtol=1e-6)  # fmt: skip
    torch.testing.assert_close(
        jp.component_mean, out.component_mean.double() * std + mean,
        atol=1e-6, rtol=1e-6,
    )  # fmt: skip
    rows = sample_joint_rows(jp, t, 5, torch.Generator().manual_seed(0))
    raw = joint_log_density(jp, rows)  # [Q, n_t, S]
    standardised = joint_log_density(
        joint_prediction_from(out), (rows - mean) / std
    )
    log_std = torch.stack([
        std.log()[torch.arange(D) != Tq].sum() for Tq in T
    ])  # fmt: skip
    torch.testing.assert_close(raw, standardised - log_std[:, None, None],
                               atol=1e-6, rtol=0)  # fmt: skip
    # independent MultivariateNormal reference on the raw parameters
    for q, Tq in enumerate(T):
        keep = torch.tensor([j for j in range(D) if j != Tq])
        for i in range(len(t)):
            parts = [
                torch.distributions.MultivariateNormal(
                    jp.component_mean[q, i, k, keep],
                    dense_covariance(jp)[q, i, k][keep][:, keep],
                ).log_prob(rows[q, i][:, keep])
                + jp.log_weights[q, i, k]
                for k in range(jp.log_weights.shape[-1])
            ]
            torch.testing.assert_close(
                raw[q, i], torch.logsumexp(torch.stack(parts), 0), atol=1e-6,
                rtol=0,
            )  # fmt: skip


# --------------------------------------------------------------------------- #
# 12, 14. Configs and smoke data
# --------------------------------------------------------------------------- #
NEW_CONFIGS = [
    "smoke_d5.yaml",
    "smoke_d5_model_joint.yaml",
    "oracle_d5_model_m_layerwise_stats_shared_joint.yaml",
]


@pytest.mark.parametrize("name", NEW_CONFIGS)
def test_new_configs_load(name):
    assert load_config_mapping(CONFIGS / name)


@pytest.mark.parametrize(
    "name",
    ["smoke_d5_model_joint.yaml",
     "oracle_d5_model_m_layerwise_stats_shared_joint.yaml"],
)  # fmt: skip
def test_joint_model_configs_build_five_variable_models(name):
    model = build_model(str(CONFIGS / name), "cpu", num_nodes=5)
    assert type(model) is JointLayerwiseMixedContextMoGModel
    assert model.encoder.share_intervention_encoder is True
    assert model.num_nodes == 5


def test_large_joint_config_rank():
    m = load_config_mapping(
        CONFIGS / "oracle_d5_model_m_layerwise_stats_shared_joint.yaml"
    )
    assert m["joint_covariance_rank"] == 8 and m["prediction_head"] == "joint"
    model = build_model(
        str(CONFIGS / "oracle_d5_model_m_layerwise_stats_shared_joint.yaml"),
        "cpu",
        num_nodes=5,
    )
    assert model.predictor.covariance_rank == 8


def test_smoke_d5_data_generates_a_grouped_batch():
    cfg = load_config(CONFIGS / "smoke_d5.yaml")
    assert list(cfg.num_variables) == [5]
    tasks = [
        generate_task(cfg, s, num_variables=5, n_ctx=80, n_groups=5)
        for s in range(2)
    ]
    batch = collate_tasks(tasks)
    prep = prepare_mixed_batch(batch, torch.float32, True, draw_outcome=False)
    assert prep["context"].shape[2] == 5 and prep["G"] == 5


def test_smoke_d5_joint_trains_one_step(tmp_path):
    args = parse_args(
        [
            "--data_config", str(CONFIGS / "smoke_d5.yaml"),
            "--model_config", str(CONFIGS / "smoke_d5_model_joint.yaml"),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", "2", "--iterations_per_epoch", "1",
            "--epochs", "1", "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--results_path", str(tmp_path), "--run_name", "r",
        ]
    )  # fmt: skip
    trainer, _ = build_trainer(args, load_config(CONFIGS / "smoke_d5.yaml"))
    assert type(trainer.model) is JointLayerwiseMixedContextMoGModel
    trainer.train()
    assert trainer.global_step == 1
    resolved = yaml.safe_load(
        (tmp_path / "results" / "r" / "resolved_model_config.yaml").read_text()
    )
    assert resolved["prediction_head"] == "joint"
    assert resolved["model_class"] == "JointLayerwiseMixedContextMoGModel"
    assert resolved["joint_covariance_rank"] == 8


# --------------------------------------------------------------------------- #
# 13, 15, 16, 19. Numerics
# --------------------------------------------------------------------------- #
def _stress_output(dtype, B=2, Nq=3, K=2, R=8, factor_size=2.0, seed=0,
                   diagonal=1e-4):  # fmt: skip
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(B, D, dtype=torch.bool)
    mask[0, 1] = mask[1, 3] = True
    return JointModelOutput(
        component_mean=torch.randn(B, Nq, K, D, generator=g).to(dtype),
        covariance_factor=(factor_size * torch.randn(B, Nq, K, D, R,
                                                     generator=g)).to(dtype),
        covariance_diagonal=torch.full((B, Nq, K, D), diagonal).to(dtype),
        mixture_logits=torch.randn(B, Nq, K, generator=g).to(dtype),
        intervened_mask=mask,
    )  # fmt: skip


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_numerical_precision_at_the_floor(dtype):
    out = _stress_output(dtype)
    g = torch.Generator().manual_seed(9)
    # targets near the component means, so the density is not negligible
    noise = 0.1 * torch.randn(2, 3, D, generator=g, dtype=torch.float64)
    target = (out.component_mean[:, :, 0].double() + noise).to(dtype)
    # reference: the same (possibly float32) values, evaluated in float64
    reference = brute_force_log_density(out, target.double())
    got = joint_mixture_log_density(out, target)
    torch.testing.assert_close(got, reference, atol=1e-6, rtol=0)


def _gradients(out, target):
    tensors = {
        name: getattr(out, name).detach().clone().requires_grad_(True)
        for name in ("component_mean", "covariance_factor",
                     "covariance_diagonal", "mixture_logits")
    }  # fmt: skip
    leaf = JointModelOutput(**tensors, intervened_mask=out.intervened_mask)
    loss = JointMixtureGaussianLoss(
        mean_loss_across_samples=True, reduce="mean"
    ).calculate_loss(leaf, target)
    loss.backward()
    return loss, {name: t.grad for name, t in tensors.items()}


@pytest.mark.parametrize(
    "case", ["floor", "dependent_columns", "large_factor", "logit_gap"]
)
def test_gradients_finite(case):
    out = _stress_output(torch.float64, factor_size=1.0)
    if case == "dependent_columns":
        base = out.covariance_factor[..., :1]
        out.covariance_factor = base + 1e-6 * out.covariance_factor
    elif case == "large_factor":
        out.covariance_factor = 100.0 * out.covariance_factor
    elif case == "logit_gap":
        out.mixture_logits = out.mixture_logits.clone()
        out.mixture_logits[..., 0] += 50.0
    target = out.component_mean[:, :, 0] + 0.1
    loss, grads = _gradients(out, target)
    assert torch.isfinite(loss)
    for name, grad in grads.items():
        assert grad is not None and torch.isfinite(grad).all(), name
    # exactly zero gradient for the outputs of intervened variables
    mask = out.intervened_mask[:, None, None, :]
    mean_mask = mask.expand_as(out.component_mean)
    factor_mask = mask[..., None].expand_as(out.covariance_factor)
    assert (grads["component_mean"][mean_mask] == 0).all()
    assert (grads["covariance_diagonal"][mean_mask] == 0).all()
    assert (grads["covariance_factor"][factor_mask] == 0).all()


def test_float32_outputs_keep_the_graph():
    out = _stress_output(torch.float32, factor_size=1.0, diagonal=0.5)
    target = out.component_mean[:, :, 0].double() + 0.1
    _, grads = _gradients(out, target)
    for name, grad in grads.items():
        assert grad.dtype == torch.float32, name
        assert grad.abs().sum() > 0, name


@pytest.mark.parametrize("cls, layers", CASES, ids=CASE_IDS)
def test_gradients_reach_the_encoder(cls, layers):
    model = build(cls, layers).train()
    inp = joint_inputs()
    out = model(**inp)
    loss = model.calculate_loss(out, inp["target_data"])
    loss.backward()
    encoder_grads = [
        p.grad for p in model.encoder.parameters() if p.grad is not None
    ]
    assert encoder_grads and any(g.abs().sum() > 0 for g in encoder_grads)


def test_cholesky_failure_gives_non_finite_loss():
    out = _stress_output(torch.float64, factor_size=0.0, diagonal=-1.0)
    target = torch.zeros(2, 3, D, dtype=torch.float64)
    loss = JointMixtureGaussianLoss(
        mean_loss_across_samples=True, reduce="mean"
    ).calculate_loss(out, target)  # must not raise
    assert not torch.isfinite(loss)


@pytest.mark.parametrize("cls, layers", CASES, ids=CASE_IDS)
def test_initialisation_positive_definite_and_finite(cls, layers):
    model = build(cls, layers)
    inp = joint_inputs()
    with torch.no_grad():
        out = model(**inp)
        cov = dense_covariance(joint_prediction_from(out)).double()
        _, info = torch.linalg.cholesky_ex(cov)
        assert (info == 0).all()
        loss = model.calculate_loss(out, inp["target_data"])
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# 20. Config validation
# --------------------------------------------------------------------------- #
def _cli_args(tmp_path, model_config, *extra):
    return parse_args(
        [
            "--data_config", str(CONFIGS / "smoke_d3.yaml"),
            "--model_config", str(model_config),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", "2", "--iterations_per_epoch", "1",
            "--epochs", "1", "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--results_path", str(tmp_path), "--run_name", "r", *extra,
        ]
    )  # fmt: skip


def _build(tmp_path, model_config, *extra):
    args = _cli_args(tmp_path, model_config, *extra)
    return build_trainer(args, load_config(CONFIGS / "smoke_d3.yaml"))


@pytest.mark.parametrize(
    "key, value",
    [("joint_covariance_rank", 4), ("joint_minimum_variance", 0.01),
     ("joint_head_node_attention_layers", 1)],
)  # fmt: skip
def test_joint_key_with_marginal_head_raises(tmp_path, key, value):
    marginal = _model_yaml(tmp_path, **{key: value})
    with pytest.raises(ValueError, match="only apply to the joint"):
        _build(tmp_path, marginal)  # from the YAML
    with pytest.raises(ValueError, match="only apply to the joint"):
        build_model(str(marginal), "cpu", num_nodes=3)
    plain = _model_yaml(tmp_path)
    flag = "--" + key
    with pytest.raises(ValueError, match="only apply to the joint"):
        _build(tmp_path, plain, flag, str(value))  # from the command line


def test_marginal_without_joint_keys_unchanged(tmp_path):
    _, config = _build(tmp_path, _model_yaml(tmp_path))
    assert config.prediction_head == "marginal"
    resolved = yaml.safe_load(
        (tmp_path / "results" / "r" / "resolved_model_config.yaml").read_text()
    )
    assert "prediction_head" not in resolved
    assert not any(k.startswith("joint_") for k in resolved)


@pytest.mark.parametrize(
    "settings, match",
    [
        ({"joint_covariance_rank": 0}, "joint_covariance_rank"),
        ({"joint_covariance_rank": 2.5}, "joint_covariance_rank"),
        ({"joint_minimum_variance": 0.0}, "joint_minimum_variance"),
        ({"joint_minimum_variance": -1e-3}, "joint_minimum_variance"),
        ({"joint_minimum_variance": float("inf")}, "joint_minimum_variance"),
        ({"joint_minimum_variance": float("nan")}, "joint_minimum_variance"),
        ({"joint_head_node_attention_layers": -1}, "node_attention"),
    ],
)
def test_invalid_joint_values_raise(settings, match):
    with pytest.raises(ValueError, match=match):
        check_joint_head_settings(settings, "joint", 3)


def test_joint_needs_mixture_components():
    with pytest.raises(ValueError, match="num_mixture_components"):
        check_joint_head_settings({}, "joint", 0)


def test_invalid_rank_from_yaml_raises(tmp_path):
    path = _model_yaml(tmp_path, prediction_head="joint",
                       joint_covariance_rank=0)  # fmt: skip
    with pytest.raises(ValueError, match="joint_covariance_rank"):
        _build(tmp_path, path)
    with pytest.raises(ValueError, match="joint_covariance_rank"):
        build_model(str(path), "cpu", num_nodes=3)


def test_prediction_head_disagreement_raises(tmp_path):
    path = _model_yaml(tmp_path, prediction_head="joint")
    with pytest.raises(ValueError, match="contradicts"):
        _build(tmp_path, path, "--prediction_head", "marginal")


def test_warm_starts_rejected_for_joint(tmp_path):
    path = _model_yaml(tmp_path, prediction_head="joint")
    for option in ("--init_from_upstream", "--init_from_r1"):
        with pytest.raises(ValueError, match="joint prediction head"):
            _build(tmp_path, path, option, "missing.pt")


# --------------------------------------------------------------------------- #
# 22. Direct evaluation of the joint
# --------------------------------------------------------------------------- #
def _correlated_scm() -> LinearSCM:
    """0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3, 3 -> 4: under do(X_0) the outcomes
    stay correlated through 1, 2 -> 3 -> 4."""
    B = np.zeros((D, D))
    B[0, 1], B[0, 2], B[1, 3], B[2, 3], B[3, 4] = 1.2, -0.8, 0.9, 0.7, 1.5
    return LinearSCM(B=B, s2=np.array([1.0, 0.5, 0.7, 0.4, 0.6]))


def test_do_joint_dist_agrees_with_do_dist():
    scm = _correlated_scm()
    for T in range(D):
        slopes, cov = scm.do_joint_dist(T)
        assert slopes[T] == pytest.approx(1.0)
        assert (cov[T] == 0).all() and (cov[:, T] == 0).all()
        for Y in range(D):
            if Y == T:
                continue
            slope, var = scm.do_dist(T, Y)
            assert slopes[Y] == pytest.approx(slope)
            assert cov[Y, Y] == pytest.approx(var)


def _truth_as_prediction(slopes, cov, T, grid, eps=1e-6):
    """A single-component JointPrediction equal to the truth."""
    keep = [j for j in range(D) if j != T]
    factor = np.zeros((D, D))
    factor[np.ix_(keep, keep)] = np.linalg.cholesky(
        cov[np.ix_(keep, keep)] - eps * np.eye(len(keep))
    )
    n_t = len(grid)
    diagonal = np.full(D, eps)
    mask = torch.zeros(1, D, dtype=torch.bool)
    mask[0, T] = True
    return JointPrediction(
        component_mean=torch.tensor(slopes[None, :] * grid[:, None])[None, :, None],
        covariance_factor=torch.tensor(factor).expand(1, n_t, 1, D, D).clone(),
        covariance_diagonal=torch.tensor(diagonal).expand(1, n_t, 1, D).clone(),
        log_weights=torch.zeros(1, n_t, 1, dtype=torch.float64),
        intervened_mask=mask,
    )  # fmt: skip


def test_joint_metrics_against_the_truth():
    scm = _correlated_scm()
    T = 0
    slopes, cov = scm.do_joint_dist(T)
    grid = np.array([-1.0, 0.0, 2.0])
    g = torch.tensor(grid, dtype=torch.float64)
    rows_np = sample_truth_joint(slopes, cov, T, grid, 4000,
                                 np.random.default_rng(0))  # fmt: skip
    assert (rows_np[..., T] == grid[:, None]).all()
    rows = torch.tensor(rows_np)
    lp_truth = truth_joint_logpdf(slopes, cov, T, g, rows)
    jp = _truth_as_prediction(slopes, cov, T, grid)
    lp_model = joint_log_density(jp, rows[None])[0]
    kl_joint = kl_truth_joint_model(lp_truth, lp_model)
    assert np.abs(kl_joint).max() < 1e-4
    _, model_cov = joint_mixture_mean_and_covariance(jp)
    assert correlation_error(model_cov[0].numpy(), cov, T).max() < 1e-4
    marginals = {}
    for j in range(D):
        if j != T:
            mu, sd, logw = joint_marginal_mixture(jp, j)
            marginals[j] = (mu[0], sd[0], logw[0])
    kl_product = kl_truth_product_of_marginals(lp_truth, marginals, rows)
    # correlated outcomes: ignoring the dependence costs a clear margin
    assert (kl_product > kl_joint + 0.05).all()


@pytest.mark.parametrize("cls, layers", CASES, ids=CASE_IDS)
def test_kshot_prior_joint_runs(cls, layers):
    from CITNP.evaluation.experiments import run_kshot_prior_joint

    cfg = MixedRegimeConfig(**SMALL_PRIOR)
    model = build(cls, layers, num_nodes=3)
    df = run_kshot_prior_joint(
        model, cfg, D=3, device="cpu", k_list=[0, 1], n_tasks=1, n_t=3,
        n_y=8,
    )  # fmt: skip
    expected = {
        "kl_truth_model_joint", "kl_truth_model_product",
        "kl_dependence_gain", "correlation_error",
        "truth_mean_abs_correlation",
    }  # fmt: skip
    assert expected <= set(df.columns)
    assert set(df["design"]) == {"obs", "random", "do(T)", "do(other)",
                                 "observational"}  # fmt: skip
    assert np.isfinite(df["kl_truth_model_joint"]).all()


def test_kshot_prior_joint_rejects_marginal_models():
    from CITNP.evaluation.experiments import run_kshot_prior_joint

    cfg = MixedRegimeConfig(**SMALL_PRIOR)
    marginal = INTERVENTION_ENCODINGS["layerwise"](tiny_config())
    with pytest.raises(ValueError, match="joint models only"):
        run_kshot_prior_joint(marginal, cfg, D=3, device="cpu", k_list=[0])


def test_kshot_prior_joint_command(tmp_path):
    from CITNP.evaluation.__main__ import main

    model = build(JointLayerwiseMixedContextMoGModel, num_nodes=3)
    ckpt = tmp_path / "joint.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    model_yaml = _model_yaml(tmp_path, prediction_head="joint",
                             joint_covariance_rank=RANK)  # fmt: skip
    out = tmp_path / "kpj.csv"
    data = str(tmp_path / "prior.yaml")
    prior = {k: list(v) if isinstance(v, tuple) else v
             for k, v in SMALL_PRIOR.items()}  # fmt: skip
    Path(data).write_text(yaml.safe_dump(prior))
    main(["kshot-prior-joint", "--ckpt", str(ckpt),
          "--model_config", str(model_yaml), "--data_config", data,
          "--n_tasks", "1", "--k_list", "0,1", "--n_t", "3", "--n_y", "8",
          "--device", "cpu", "--out", str(out)])  # fmt: skip
    assert out.is_file() and (tmp_path / "kpj_D3_joint.png").is_file()
    assert (tmp_path / "kpj_meta.json").is_file()
    marginal_yaml = tmp_path / "marginal.yaml"
    marginal_yaml.write_text(yaml.safe_dump(
        {k: v for k, v in yaml.safe_load(model_yaml.read_text()).items()
         if k not in ("prediction_head", "joint_covariance_rank")}
    ))  # fmt: skip
    ckpt_marginal = tmp_path / "marginal.pt"
    torch.manual_seed(0)
    torch.save(
        {"model_state_dict": INTERVENTION_ENCODINGS["layerwise"](
            tiny_config()).state_dict()},
        ckpt_marginal,
    )  # fmt: skip
    with pytest.raises(SystemExit):
        main(["kshot-prior-joint", "--ckpt", str(ckpt_marginal),
              "--model_config", str(marginal_yaml), "--data_config", data,
              "--n_tasks", "1", "--k_list", "0", "--device", "cpu",
              "--out", str(tmp_path / "m.csv")])  # fmt: skip


# --------------------------------------------------------------------------- #
# 23. One conditional per group: query rows are independent
# --------------------------------------------------------------------------- #
def _outputs_of(out, rows):
    return [
        getattr(out, name)[:, rows]
        for name in ("component_mean", "covariance_factor",
                     "covariance_diagonal", "mixture_logits")
    ]  # fmt: skip


@pytest.mark.parametrize("cls, layers", CASES, ids=CASE_IDS)
def test_query_rows_are_independent(cls, layers):
    model = perturb(build(cls, layers, double=True))
    inp = joint_inputs(Nq=5, dtype=torch.float64)
    target = inp["target_data"].clone()
    target[:, 3] = target[:, 0]  # rows 0 and 3 share their do-value
    inp["target_data"] = target
    with torch.no_grad():
        full = model(**inp)
        for a, b in zip(_outputs_of(full, [0]), _outputs_of(full, [3])):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
        variants = {
            "removed": target[:, :1],
            "changed": torch.cat([target[:, :1], 5.0 + target[:, 1:]], dim=1),
            "reordered": target.flip(1),  # row 0 becomes the last row
        }
        for name, other in variants.items():
            out = model(**{**inp, "target_data": other})
            row = other.shape[1] - 1 if name == "reordered" else 0
            for a, b in zip(_outputs_of(full, [0]), _outputs_of(out, [row])):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=0, msg=name)
