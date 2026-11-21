"""CITNP integration of causal_active_learning: the model adapter, model_api
additions, the linear SCM environment, the support reader and the command."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_active_learning.acquisitions import CandidateExperiment
from causal_active_learning.config import ACQUISITIONS
from causal_active_learning.context import ExperimentContext
from causal_active_learning.integrations.citnp.environment import (
    LinearSCMEnvironment,
)
from causal_active_learning.integrations.citnp.model import (
    ModelApiPredictiveModel,
)
from causal_active_learning.integrations.citnp.support import (
    training_support_from_data_config,
)
from causal_active_learning.loop import (
    ActiveLearningGenerators,
    run_active_learning,
)
from causal_active_learning.selection import select_next_experiment
from causal_active_learning.support_check import TrainingSupport
from CITNP.evaluation.model_api import (
    JointPrediction,
    JointSampleNoise,
    _batched_model_inputs,
    _model_inputs,
    draw_joint_sample_noise,
    predict_joint_mixture,
    predict_joint_mixture_batched,
    predict_mixture,
    predict_mixture_batched,
    sample_joint_rows,
)
from CITNP.evaluation.oracle import LinearSCM
from CITNP.trainer.mixed_context_trainer import _normalisation_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import PSI, empty_context
from test_causal_active_learning_selection import cheap_config
from test_joint_prediction import (
    RANK,
    JointLayerwiseMixedContextMoGModel,
    _model_yaml,
    build,
    perturb,
)
from test_mixed_context_model import tiny_config

DOUBLE = torch.float64
REPO = Path(__file__).resolve().parents[1]
D5_CONFIG = REPO / "CITNP/experiments/configs/oracle_d5_iw_a15_sc.yaml"


def tiny_joint_model(seed=0, num_nodes=5):
    torch.manual_seed(seed)
    return perturb(
        build(JointLayerwiseMixedContextMoGModel, seed=seed, num_nodes=num_nodes)
    )


def example_environment():
    B = np.zeros((5, 5))
    B[0, 1], B[1, 2], B[2, 4], B[0, 3], B[3, 4], B[0, 2] = (
        1,
        0.8,
        1.2,
        -0.7,
        0.5,
        0.4,
    )
    return LinearSCMEnvironment(
        LinearSCM(B, np.full(5, 0.5)), {"kind": "normal", "scale": 1.5}, B != 0
    )


# --------------------------------------------------------------------------- #
# Fantasy contexts: model inputs and standardisation
# --------------------------------------------------------------------------- #
def test_fantasy_context_inputs_equal_building_from_scratch():
    context = empty_context(rows=15, D=5)
    rows = torch.randn(3, 5, dtype=DOUBLE)
    for node in (3, None):
        augmented = context.append(rows, node)
        built, mean_b, std_b = _batched_model_inputs(
            augmented.values[None], augmented.intervened[None],
            torch.tensor([0]), torch.tensor([1]), torch.zeros(1, 2),
            normalise=True,
        )  # fmt: skip
        scratch, mean_s, std_s = _model_inputs(
            augmented.values.numpy(), augmented.intervened.numpy(), 0, 1,
            np.zeros(2), zero_flags=False, normalise=True,
        )  # fmt: skip
        assert torch.allclose(mean_b[0], mean_s) and torch.allclose(
            std_b[0], std_s
        )
        for key in ("context_data", "context_scale", "context_int_mask"):
            assert torch.allclose(built[key].float(), scratch[key].float()), key
        # the core's observational statistics are the training code's
        mean, std = augmented.observational_mean_and_standard_deviation()
        ref_mean, ref_std = _normalisation_stats(
            augmented.values[None].float(), augmented.intervened[None]
        )
        assert torch.allclose(mean.float(), ref_mean[0], atol=1e-5)
        assert torch.allclose(std.float(), ref_std[0], atol=1e-5)


# --------------------------------------------------------------------------- #
# Fantasies from the adapter
# --------------------------------------------------------------------------- #
def test_fantasies_match_joint_samples_and_fix_the_target():
    model = tiny_joint_model()
    adapter = ModelApiPredictiveModel(model, "cpu")
    context = empty_context(rows=30, D=5)
    g = torch.Generator().manual_seed(1)
    candidates = [CandidateExperiment(1, 0.7), CandidateExperiment(3, -1.2)]
    noise = [adapter.draw_fantasy_noise(5, 6, 1, g) for _ in candidates]
    rows = adapter.sample_fantasies([context] * 2, candidates, noise)
    assert rows.shape == (2, 6, 1, 5)
    for c, cand in enumerate(candidates):
        assert torch.all(
            rows[c, :, 0, cand.target_node] == cand.intervention_value
        )
        joint = predict_joint_mixture(
            model, context.values.numpy(), context.intervened.numpy(),
            cand.target_node, np.array([cand.intervention_value]), device="cpu",
        )  # fmt: skip
        reference = sample_joint_rows(
            joint, np.array([cand.intervention_value]), 6,
            noise=noise[c].row(0),
        )[0, 0]  # fmt: skip
        assert torch.allclose(rows[c, :, 0], reference, atol=1e-4)


@pytest.mark.parametrize("sampling", ["autoregressive", "independent"])
def test_multi_row_fantasies(sampling):
    adapter = ModelApiPredictiveModel(
        tiny_joint_model(), "cpu", fantasy_row_sampling=sampling
    )
    g = torch.Generator().manual_seed(0)
    R = 3
    candidates = [CandidateExperiment(1, 1.0, R), CandidateExperiment(3, 0.5, R)]
    noise = [adapter.draw_fantasy_noise(5, 4, R, g) for _ in candidates]
    rows = adapter.sample_fantasies([empty_context(D=5)] * 2, candidates, noise)
    assert rows.shape == (2, 4, R, 5)
    assert torch.all(rows[0, :, :, 1] == 1.0) and torch.all(
        rows[1, :, :, 3] == 0.5
    )
    assert torch.isfinite(rows).all()


def test_observational_fantasies_are_not_available():
    adapter = ModelApiPredictiveModel(tiny_joint_model(), "cpu")
    assert adapter.supports_observational_prediction is False
    noise = [adapter.draw_fantasy_noise(5, 4, 1, torch.Generator())]
    with pytest.raises(NotImplementedError):
        adapter.sample_fantasies(
            [empty_context(D=5)], [CandidateExperiment(None, None)], noise
        )
    with pytest.raises(ValueError):
        from CITNP.models.mixed_context_model import INTERVENTION_ENCODINGS

        ModelApiPredictiveModel(
            INTERVENTION_ENCODINGS["layerwise"](tiny_config(5)), "cpu"
        )


def _reference_sample_joint_rows(jp, t, num_samples, generator):
    """sample_joint_rows as it was before the noise argument existed."""
    Q, n_t, K, D = jp.component_mean.shape
    R = jp.covariance_factor.shape[-1]
    dtype, device = jp.component_mean.dtype, jp.component_mean.device
    weights = jp.log_weights.exp().reshape(-1, K)
    component = torch.multinomial(
        weights, num_samples, replacement=True, generator=generator
    )
    component = component.reshape(Q, n_t, num_samples)
    index = component[..., None].expand(Q, n_t, num_samples, D)
    mu = jp.component_mean.gather(2, index)
    diagonal = jp.covariance_diagonal.gather(2, index)
    factor = jp.covariance_factor.gather(
        2, index[..., None].expand(Q, n_t, num_samples, D, R)
    )
    z = torch.randn(
        Q, n_t, num_samples, R, generator=generator, dtype=dtype, device=device
    )
    e = torch.randn(
        Q, n_t, num_samples, D, generator=generator, dtype=dtype, device=device
    )
    rows = mu + (factor @ z[..., None]).squeeze(-1) + diagonal.sqrt() * e
    do_values = torch.as_tensor(np.asarray(t), dtype=dtype, device=device)
    intervened = jp.intervened_mask[:, None, None, :]
    return torch.where(intervened, do_values[None, :, None, None], rows)


def _random_joint(Q=2, n_t=3, K=3, D=4, R=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(Q, D, dtype=torch.bool)
    mask[torch.arange(Q), torch.arange(Q) % D] = True
    return JointPrediction(
        component_mean=torch.randn(Q, n_t, K, D, generator=g, dtype=DOUBLE),
        covariance_factor=torch.randn(Q, n_t, K, D, R, generator=g, dtype=DOUBLE),
        covariance_diagonal=torch.rand(Q, n_t, K, D, generator=g, dtype=DOUBLE)
        + 0.1,
        log_weights=torch.log_softmax(
            torch.randn(Q, n_t, K, generator=g, dtype=DOUBLE), -1
        ),
        intervened_mask=mask,
    )


def test_sample_joint_rows_without_noise_is_unchanged():
    jp = _random_joint()
    t = np.array([-1.0, 0.0, 2.0])
    new = sample_joint_rows(jp, t, 50, torch.Generator().manual_seed(7))
    old = _reference_sample_joint_rows(
        jp, t, 50, torch.Generator().manual_seed(7)
    )
    assert torch.equal(new, old)


def test_rows_are_continuous_in_the_do_value_with_fixed_noise():
    noise = draw_joint_sample_noise(16, 2, 4, torch.Generator().manual_seed(0))
    slopes = torch.tensor([0.0, 1.0, -0.5, 2.0], dtype=DOUBLE)

    def rows_at(v):
        jp = JointPrediction(
            component_mean=(slopes * v)[None, None, None],
            covariance_factor=torch.full((1, 1, 1, 4, 2), 0.3, dtype=DOUBLE),
            covariance_diagonal=torch.full((1, 1, 1, 4), 0.5, dtype=DOUBLE)
            * (1 + v**2),
            log_weights=torch.zeros(1, 1, 1, dtype=DOUBLE),
            intervened_mask=torch.tensor([[True, False, False, False]]),
        )
        return sample_joint_rows(jp, np.array([v]), 16, noise=noise)[0, 0]

    a, b = rows_at(0.7), rows_at(0.7 + 1e-6)
    assert (a - b).abs().max() < 1e-5
    assert torch.all(a[:, 0] == 0.7)
    # the component is chosen by inverse cumulative distribution
    jp = _random_joint(Q=1, n_t=1)
    rows = sample_joint_rows(jp, np.array([0.0]), 16, noise=JointSampleNoise(
        component_uniform=torch.full((16,), 1e-9, dtype=DOUBLE),
        factor_normal=torch.zeros(16, 2, dtype=DOUBLE),
        residual_normal=torch.zeros(16, 4, dtype=DOUBLE),
    ))  # fmt: skip
    expected = jp.component_mean[0, 0, 0].clone()
    expected[0] = 0.0
    assert torch.allclose(rows[0, 0], expected.expand(16, 4))


# --------------------------------------------------------------------------- #
def test_batched_model_api_equals_looping():
    from CITNP.models.mixed_context_model import INTERVENTION_ENCODINGS

    joint = tiny_joint_model()
    marginal = INTERVENTION_ENCODINGS["layerwise"](tiny_config(5)).eval()
    B, N, D = 3, 25, 5
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, N, D, generator=g, dtype=DOUBLE)
    m = torch.zeros(B, N, D, dtype=torch.bool)
    m[:, :4, 1] = True
    T = torch.tensor([0, 2, 4])
    t = torch.randn(B, 2, generator=g, dtype=DOUBLE)
    batched = predict_joint_mixture_batched(joint, x, m, T, t, device="cpu")
    for b in range(B):
        single = predict_joint_mixture(
            joint, x[b].numpy(), m[b].numpy(), int(T[b]), t[b].numpy(),
            device="cpu",
        )  # fmt: skip
        for name in ("component_mean", "covariance_factor",
                     "covariance_diagonal", "log_weights"):  # fmt: skip
            assert torch.allclose(
                getattr(batched, name)[b], getattr(single, name)[0], atol=1e-5
            ), name
        assert torch.equal(batched.intervened_mask[b], single.intervened_mask[0])
    grid = np.linspace(-1, 1, 4)
    for model in (joint, marginal):
        mu, sd, lw = predict_mixture_batched(
            model, x, m, 1, 3, grid, device="cpu"
        )
        for b in range(B):
            mu1, sd1, lw1 = predict_mixture(
                model, x[b].numpy(), m[b].numpy(), 1, 3, grid, device="cpu"
            )
            assert torch.allclose(mu[b], mu1[0], atol=1e-5)
            assert torch.allclose(sd[b], sd1[0], atol=1e-5)
            assert torch.allclose(lw[b], lw1[0], atol=1e-5)


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def test_environment_rows_have_the_exact_do_value():
    env = example_environment()
    rows = env.run_experiment(
        CandidateExperiment(1, 0.37, 5), np.random.default_rng(0)
    )
    assert rows.shape == (5, 5) and torch.all(rows[:, 1] == 0.37)
    truth = env.true_query_distribution(0, 2, PSI)
    slope, variance = env.scm.do_dist(0, 2)
    assert torch.allclose(truth.mean[:, 0], slope * PSI)
    assert torch.allclose(
        truth.standard_deviation[:, 0], torch.full_like(PSI, math.sqrt(variance))
    )


@pytest.mark.slow
def test_environment_moments_match_the_interventional_joint():
    env = example_environment()
    rows = env.run_experiment(
        CandidateExperiment(3, 1.5, 200_000), np.random.default_rng(1)
    ).numpy()
    slopes, covariance = env.scm.do_joint_dist(3)
    assert np.allclose(rows.mean(0), slopes * 1.5, atol=0.01)
    assert np.allclose(np.cov(rows.T), covariance, atol=0.01)


# --------------------------------------------------------------------------- #
# End to end, support, command
# --------------------------------------------------------------------------- #
def test_end_to_end_tiny_joint_model():
    adapter = ModelApiPredictiveModel(tiny_joint_model(), "cpu")
    env = example_environment()
    context = ExperimentContext(
        env.sample_observational(30, np.random.default_rng(0)),
        torch.zeros(30, 5, dtype=torch.bool),
    )
    for optimiser in ("bayesian_optimisation", "grid"):
        for acquisition in ACQUISITIONS:
            config = cheap_config(
                acquisition=acquisition, value_optimiser=optimiser,
                num_fantasies=4, final_num_fantasies=4, quadrature_points=128,
                bayesian_optimisation_iterations=1, gaussian_process_restarts=1,
            )  # fmt: skip
            steps = run_active_learning(
                adapter, env, context, 2, 4, PSI, config,
                ActiveLearningGenerators.from_seed(0),
            )  # fmt: skip
            assert len(steps) == 2
            for step in steps:
                assert math.isfinite(step.kl_to_truth_after)
                for s in step.scores:
                    assert math.isfinite(s.expected_information_gain)
    with pytest.raises(NotImplementedError):
        select_next_experiment(
            adapter, context, 2, 4, PSI,
            cheap_config(include_observational_candidate=True, num_fantasies=4,
                         final_num_fantasies=4),
            ActiveLearningGenerators.from_seed(0),
        )  # fmt: skip


def test_training_support_from_the_d5_config():
    from CITNP.datasets.create_mixed_regime_chunks import load_config

    support = training_support_from_data_config(load_config(str(D5_CONFIG)), 5)
    assert support == TrainingSupport((64, 360), 33, 5)


def test_run_command(tmp_path):
    import json

    import pandas as pd

    from causal_active_learning.integrations.citnp.command import main

    ckpt = tmp_path / "joint.pt"
    torch.save({"model_state_dict": tiny_joint_model().state_dict()}, ckpt)
    model_yaml = _model_yaml(
        tmp_path, prediction_head="joint", joint_covariance_rank=RANK
    )
    out = tmp_path / "al"
    main([
        "run", "--ckpt", str(ckpt), "--model_config", str(model_yaml),
        "--data_config", str(D5_CONFIG), "--environment", "example",
        "--fixed_graph_edges", "0->1,1->2,2->4,0->3,3->4,0->2",
        "--treatment", "2", "--outcome", "4",
        "--strategies", "expected_information_gain,random",
        "--value_optimisers", "bayesian_optimisation,grid",
        "--experiment_budget", "1", "--seeds", "0", "--num_fantasies", "4",
        "--final_num_fantasies", "4", "--quadrature_points", "64",
        "--bayesian_optimisation_iterations", "1",
        "--gaussian_process_restarts", "1", "--dense_value_diagnostic",
        "--dense_value_diagnostic_points", "3", "--device", "cpu",
        "--out", str(out),
    ])  # fmt: skip
    for name in ("steps", "scores", "bayesian_optimisation_trace", "surrogate",
                 "dense_value_diagnostic"):  # fmt: skip
        assert (tmp_path / f"al_{name}.csv").is_file(), name
    assert (tmp_path / "al_kl_to_truth.png").is_file()
    meta = json.loads((tmp_path / "al_meta.json").read_text())
    assert meta["tasks"][0]["treatment"] == 2
    steps = pd.read_csv(tmp_path / "al_steps.csv")
    assert set(steps.strategy) == {
        "expected_information_gain/bayesian_optimisation",
        "expected_information_gain/grid", "random",
    }  # fmt: skip
