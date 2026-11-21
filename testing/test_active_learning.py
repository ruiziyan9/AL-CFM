"""Tests for the active learning loop (CITNP/active_learning, plan section 9).

Fast, CPU, float64 where it matters. Closed-form checks use stub predictive
models (`PredictiveModel` protocol); the BO tests pass stub `evaluate`
functions directly; the end-to-end test uses a tiny untrained joint model.
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from CITNP.active_learning import (
    ActiveLearningConfig,
    ActiveLearningGenerators,
    BayesianOptimisationValueOptimiser,
    CandidateExperiment,
    CandidateScore,
    ExperimentContext,
    GridValueOptimiser,
    InformationGainEstimate,
    LinearSCMEnvironment,
    ModelApiPredictiveModel,
    QueryMixture,
    TrainingSupport,
    ValueDomain,
    append_rows_to_context,
    candidate_target_nodes,
    choice_is_clear,
    estimate_expected_information_gain,
    run_active_learning,
    sample_fantasy_rows,
    select_next_experiment,
)
from CITNP.active_learning import loop as loop_module
from CITNP.active_learning.config import ACQUISITIONS
from CITNP.active_learning.design_optimisation import (
    ValueOptimisationResult,
    reevaluate_and_select,
)
from CITNP.active_learning.experiment import (
    ActiveLearningTask,
    make_strategies,
    run_active_learning_experiment,
)
from CITNP.active_learning.gaussian_process import (
    GaussianProcessHyperparameters,
    condition_gaussian_process,
    fit_gaussian_process,
    matern52_kernel,
    robust_cholesky,
)
from CITNP.active_learning.information_gain import (
    kl_divergence_monte_carlo,
    kl_divergence_quadrature,
)
from CITNP.active_learning.loop import draw_fantasy_noise
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_joint_prediction import (
    JointLayerwiseMixedContextMoGModel,
    build,
    perturb,
)
from test_mixed_context_model import tiny_config

DOUBLE = torch.float64


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
def _gaussian_joint(mean, variance, target_nodes):
    """One-component diagonal `JointPrediction` (Q = B, n_t = 1, R = 1) from
    means and variances [B, D]."""
    B, D = mean.shape
    mask = torch.zeros(B, D, dtype=torch.bool)
    nodes = torch.as_tensor(target_nodes, dtype=torch.long)
    valid = nodes >= 0
    mask[torch.arange(B)[valid], nodes[valid]] = True
    return JointPrediction(
        component_mean=mean[:, None, None, :].to(DOUBLE),
        covariance_factor=torch.zeros(B, 1, 1, D, 1, dtype=DOUBLE),
        covariance_diagonal=variance[:, None, None, :].to(DOUBLE),
        log_weights=torch.zeros(B, 1, 1, dtype=DOUBLE),
        intervened_mask=mask,
    )


class ConjugateStub:
    """A coherent Bayesian predictive with a closed-form EIG.

    Nodes: 0 treatment T, 1 informative node i, 2 outcome Y, 3 uninformative
    node j. The outcome responds to the treatment and to node i with the
    same unknown coefficient beta ~ N(PRIOR_MEAN, PRIOR_VARIANCE), noise
    variance NOISE:  Y | do(X_T = psi) ~ N(beta psi, NOISE). Rows intervened
    on i observe y = beta v + e and update beta (conjugate); nothing else
    does. Intervening on j leaves Y ~ N(0, 1) and teaches nothing.
    """

    PRIOR_MEAN, PRIOR_VARIANCE, NOISE = 0.5, 1.0, 0.25
    num_variables = 4
    covariance_rank = 1

    def posterior(self, values, intervened):
        """Posterior mean and variance of beta [B] from contexts [B, N, D]."""
        on_i = intervened[..., 1].to(DOUBLE)
        v, y = values[..., 1], values[..., 2]
        precision = 1 / self.PRIOR_VARIANCE + (on_i * v**2).sum(-1) / self.NOISE
        mean = (
            self.PRIOR_MEAN / self.PRIOR_VARIANCE
            + (on_i * v * y).sum(-1) / self.NOISE
        ) / precision
        return mean, 1 / precision

    def predict_query(self, values, intervened, treatment, outcome, psi):
        m, s2 = self.posterior(values, intervened)
        psi = torch.as_tensor(psi, dtype=DOUBLE)
        mean = m[:, None] * psi
        sd = torch.sqrt(s2[:, None] * psi**2 + self.NOISE)
        return QueryMixture.gaussian(mean, sd)

    def predict_intervention_joint(self, values, intervened, nodes, do_values):
        m, s2 = self.posterior(values, intervened)
        B = values.shape[0]
        mean = torch.zeros(B, 4, dtype=DOUBLE)
        variance = torch.ones(B, 4, dtype=DOUBLE)
        informative = torch.as_tensor(nodes) == 1
        v = torch.as_tensor(do_values, dtype=DOUBLE)
        mean[:, 2] = torch.where(informative, m * v, 0.0)
        variance[:, 2] = torch.where(informative, s2 * v**2 + self.NOISE, 1.0)
        return _gaussian_joint(mean, variance, nodes)

    def predict_observational_joint(self, values, intervened):
        B = values.shape[0]
        return _gaussian_joint(
            torch.zeros(B, 4, dtype=DOUBLE), torch.ones(B, 4, dtype=DOUBLE),
            torch.full((B,), -1),
        )  # fmt: skip

    def exact_information_gain(self, v, psi):
        """E_psi of -0.5 log(1 - rho^2) between Y_psi and Y_v under the
        prior (an empty context)."""
        s2, n = self.PRIOR_VARIANCE, self.NOISE
        psi = torch.as_tensor(psi, dtype=DOUBLE)
        rho2 = (s2 * psi * v) ** 2 / ((s2 * psi**2 + n) * (s2 * v**2 + n))
        return float((-0.5 * torch.log1p(-rho2)).mean())


class ContextBlindStub(ConjugateStub):
    """The query ignores the context: every EIG is zero."""

    def predict_query(self, values, intervened, treatment, outcome, psi):
        B = values.shape[0]
        psi = torch.as_tensor(psi, dtype=DOUBLE)
        return QueryMixture.gaussian(
            torch.zeros(B, len(psi), dtype=DOUBLE),
            torch.ones(B, len(psi), dtype=DOUBLE),
        )


class CountingStub(ConjugateStub):
    def __init__(self):
        self.joint_calls = 0

    def predict_intervention_joint(self, *args):
        self.joint_calls += 1
        return super().predict_intervention_joint(*args)


def _empty_context(rows=20, D=4, seed=0):
    """An observational context (enough rows for the standardisation)."""
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, D, generator=g, dtype=DOUBLE)
    return ExperimentContext(values, torch.zeros(rows, D, dtype=torch.bool))


def _cheap_config(**kw):
    settings = dict(
        experiment_budget=2, num_fantasies=8, final_num_fantasies=16,
        quadrature_points=256, bayesian_optimisation_iterations=2,
        gaussian_process_restarts=2, acquisition_grid_points=128,
    )  # fmt: skip
    settings.update(kw)
    return ActiveLearningConfig(**settings)


def _stub_environment():
    B = np.zeros((4, 4))
    B[0, 2], B[1, 2], B[3, 1] = 0.8, 0.8, 0.5
    return LinearSCMEnvironment(
        LinearSCM(B, np.full(4, 0.3)), {"kind": "normal", "scale": 1.5}, B != 0
    )


PSI = torch.linspace(-2, 2, 7, dtype=DOUBLE)


# --------------------------------------------------------------------------- #
# 1. Context augmentation
# --------------------------------------------------------------------------- #
def test_context_augmentation():
    context = _empty_context(rows=15, D=5)
    rows = torch.randn(2, 5, dtype=DOUBLE)
    values, flags = append_rows_to_context(
        context.values, context.intervened, rows, 3
    )
    assert values.shape == (17, 5) and flags.shape == (17, 5)
    assert torch.equal(values[:15], context.values)
    assert not flags[:15].any()
    assert flags[15:, 3].all() and flags[15:].sum() == 2
    _, obs_flags = append_rows_to_context(
        context.values, context.intervened, rows, None
    )
    assert not obs_flags.any()
    # batched form, one node per context, -1 = observational
    batch_values = context.values[None].expand(2, -1, -1)
    batch_flags = context.intervened[None].expand(2, -1, -1)
    _, per_context = append_rows_to_context(
        batch_values, batch_flags, rows[None].expand(2, -1, -1),
        torch.tensor([1, -1]),
    )  # fmt: skip
    assert per_context[0, 15:, 1].all() and per_context[0].sum() == 2
    assert not per_context[1].any()
    # the model inputs of the augmented context (standardisation, scale
    # features) equal those built from scratch by the existing path
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
    # observational rows change the standardisation, interventional do not
    _, m_int, _ = _model_inputs(
        context.append(rows, 3).values.numpy(),
        context.append(rows, 3).intervened.numpy(), 0, 1, np.zeros(1),
        zero_flags=False, normalise=True,
    )  # fmt: skip
    _, m_obs, _ = _model_inputs(
        context.append(rows, None).values.numpy(),
        context.append(rows, None).intervened.numpy(), 0, 1, np.zeros(1),
        zero_flags=False, normalise=True,
    )  # fmt: skip
    _, m_0, _ = _model_inputs(
        context.values.numpy(), context.intervened.numpy(), 0, 1, np.zeros(1),
        zero_flags=False, normalise=True,
    )  # fmt: skip
    assert torch.allclose(m_int, m_0) and not torch.allclose(m_obs, m_0)


# --------------------------------------------------------------------------- #
# 2. Fantasy rows
# --------------------------------------------------------------------------- #
def _tiny_joint_model(seed=0):
    torch.manual_seed(seed)
    return perturb(build(JointLayerwiseMixedContextMoGModel, seed=seed))


def test_fantasy_rows_match_joint_samples_and_fix_the_target():
    model = _tiny_joint_model()
    predictive = ModelApiPredictiveModel(model, 5, "cpu")
    context = _empty_context(rows=30, D=5)
    g = torch.Generator().manual_seed(1)
    candidates = [CandidateExperiment(1, 0.7), CandidateExperiment(3, -1.2)]
    noises = [
        draw_joint_sample_noise(6, predictive.covariance_rank, 5, g)
        for _ in candidates
    ]
    rows = sample_fantasy_rows(predictive, context, candidates, noises)
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
            joint, np.array([cand.intervention_value]), 6, noise=noises[c]
        )[0, 0]
        assert torch.allclose(rows[c, :, 0], reference, atol=1e-4)


def test_autoregressive_calls_the_model_once_per_row():
    stub = CountingStub()
    context = _empty_context()
    g = torch.Generator().manual_seed(0)
    R = 3
    candidates = [CandidateExperiment(1, 1.0, R), CandidateExperiment(3, 0.5, R)]
    noises = [draw_joint_sample_noise(4, 1, 4, g, num_rows=R) for _ in candidates]
    rows = sample_fantasy_rows(stub, context, candidates, noises, batch_size=1000)
    assert rows.shape == (2, 4, R, 4)
    assert stub.joint_calls == R
    assert torch.all(rows[0, :, :, 1] == 1.0) and torch.all(
        rows[1, :, :, 3] == 0.5
    )
    stub.joint_calls = 0
    independent = sample_fantasy_rows(
        stub, context, candidates, noises, mode="independent"
    )
    assert stub.joint_calls == 1 and independent.shape == rows.shape


def test_autoregressive_rows_follow_the_updated_posterior():
    """Row 2 of a node-1 experiment is drawn from the posterior after row 1:
    with an informative first row, its mean moves."""
    stub = ConjugateStub()
    context = _empty_context()
    zero = JointSampleNoise(
        component_uniform=torch.zeros(1, 2, dtype=DOUBLE),
        factor_normal=torch.zeros(1, 2, 1, dtype=DOUBLE),
        residual_normal=torch.zeros(1, 2, 4, dtype=DOUBLE),
    )
    zero.residual_normal[0, 0, 2] = 3.0  # first outcome far above the mean
    rows = sample_fantasy_rows(
        stub, context, [CandidateExperiment(1, 1.0, 2)], [zero]
    )
    m0, _ = stub.posterior(context.values[None], context.intervened[None])
    first = context.append(rows[0, 0, :1], 1)
    m1, _ = stub.posterior(first.values[None], first.intervened[None])
    assert float(rows[0, 0, 1, 2]) == pytest.approx(float(m1[0]))
    assert float(m1[0]) > float(m0[0]) + 0.5


# --------------------------------------------------------------------------- #
# 3. The noise argument of sample_joint_rows
# --------------------------------------------------------------------------- #
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
# 4. Quadrature KL
# --------------------------------------------------------------------------- #
def test_quadrature_kl_matches_closed_form_and_monte_carlo():
    m1, s1, m2, s2 = 0.3, 0.8, -0.4, 1.7
    p = QueryMixture.gaussian(torch.tensor([m1]), torch.tensor([s1]))
    q = QueryMixture.gaussian(torch.tensor([m2]), torch.tensor([s2]))
    exact = math.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5
    assert float(kl_divergence_quadrature(p, q, 1024)) == pytest.approx(
        exact, abs=1e-6
    )
    g = torch.Generator().manual_seed(0)
    p = QueryMixture(
        torch.tensor([[-1.0, 1.5]], dtype=DOUBLE),
        torch.tensor([[0.5, 0.8]], dtype=DOUBLE),
        torch.log(torch.tensor([[0.3, 0.7]], dtype=DOUBLE)),
    )
    q = QueryMixture(
        torch.tensor([[0.0, 2.0, -2.0]], dtype=DOUBLE),
        torch.tensor([[1.0, 0.6, 1.2]], dtype=DOUBLE),
        torch.log(torch.tensor([[0.5, 0.3, 0.2]], dtype=DOUBLE)),
    )
    quadrature = float(kl_divergence_quadrature(p, q, 1024)[0])
    draws = torch.stack(
        [kl_divergence_monte_carlo(p, q, 4000, g)[0] for _ in range(10)]
    )
    error = float(draws.std() / math.sqrt(len(draws)))
    assert abs(float(draws.mean()) - quadrature) < 4 * error + 1e-3
    assert quadrature > 0


# --------------------------------------------------------------------------- #
# 5-6. Zero and known information
# --------------------------------------------------------------------------- #
def _noise_for(nodes, S, stub, seed=0, rows=1):
    return draw_fantasy_noise(
        nodes, S, stub, rows, torch.Generator().manual_seed(seed)
    )


def test_context_blind_query_gives_zero_information():
    stub = ContextBlindStub()
    candidates = [CandidateExperiment(1, 1.5), CandidateExperiment(3, -0.5)]
    estimates = estimate_expected_information_gain(
        stub, _empty_context(), 0, 2, PSI, candidates, 8,
        _noise_for([1, 3], 8, stub), _cheap_config(),
        compute_coherence_gap=True,
    )  # fmt: skip
    for e in estimates:
        assert e.expected_information_gain == 0.0
        assert e.standard_error == 0.0
        assert e.coherence_gap == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("v", [0.5, 1.5, -2.5])
def test_conjugate_information_gain_matches_closed_form(v):
    stub = ConjugateStub()
    S = 2000
    estimate = estimate_expected_information_gain(
        stub, _empty_context(), 0, 2, PSI, [CandidateExperiment(1, v)], S,
        _noise_for([1], S, stub, seed=3), _cheap_config(quadrature_points=512),
        compute_coherence_gap=True,
    )[0]  # fmt: skip
    exact = stub.exact_information_gain(v, PSI)
    assert abs(estimate.expected_information_gain - exact) < (
        4 * estimate.standard_error + 1e-3
    )
    assert estimate.coherence_gap < 2e-3
    uninformative = estimate_expected_information_gain(
        stub, _empty_context(), 0, 2, PSI, [CandidateExperiment(3, v)], 8,
        _noise_for([3], 8, stub), _cheap_config(),
    )[0]  # fmt: skip
    assert uninformative.expected_information_gain == 0.0


# --------------------------------------------------------------------------- #
# 7. Batched prediction
# --------------------------------------------------------------------------- #
def test_batched_model_api_equals_looping():
    from CITNP.models.mixed_context_model import INTERVENTION_ENCODINGS

    joint = _tiny_joint_model()
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
# 8. Gaussian process
# --------------------------------------------------------------------------- #
def _reference_posterior(x, y, noise, h, test, jitter=1e-6):
    """Dense-matrix GP posterior, written independently (1-D inputs)."""
    mean_y = y.mean()
    scale = y.std(unbiased=False)
    ys = (y - mean_y) / scale

    def k(a, b):
        r = (a[:, None] - b[None, :]).abs() / h.lengthscale
        return (
            h.output_scale
            * (1 + math.sqrt(5) * r + 5 * r**2 / 3)
            * torch.exp(-math.sqrt(5) * r)
        )

    K = k(x, x) + torch.diag(noise / scale**2) + jitter * torch.eye(len(x))
    cross = k(x, test)
    mean = h.constant_mean + cross.T @ torch.linalg.solve(K, ys - h.constant_mean)
    var = h.output_scale - (cross * torch.linalg.solve(K, cross)).sum(0)
    return mean_y + scale * mean, scale * var.clamp(min=0).sqrt()


def test_gaussian_process_against_dense_reference():
    x = torch.tensor([-2.0, -0.5, 0.3, 1.1, 2.4], dtype=DOUBLE)
    y = torch.tensor([0.1, 0.4, 0.9, 0.7, 0.2], dtype=DOUBLE)
    noise = torch.tensor([1e-3, 4e-3, 2e-3, 1e-2, 5e-3], dtype=DOUBLE)
    h = GaussianProcessHyperparameters(0.9, 1.3, 0.2)
    gp = condition_gaussian_process(x[:, None], y, noise, h)
    test = torch.linspace(-3, 3, 41, dtype=DOUBLE)
    mean, sd = gp.posterior(test[:, None])
    ref_mean, ref_sd = _reference_posterior(x, y, noise, h, test)
    assert torch.allclose(mean, ref_mean, atol=1e-8)
    assert torch.allclose(sd, ref_sd, atol=1e-8)


def test_gaussian_process_interpolates_with_tiny_noise_and_respects_bounds():
    x = torch.tensor([[-2.0], [-0.5], [0.3], [1.1], [2.4]], dtype=DOUBLE)
    y = torch.sin(x[:, 0])
    gp = fit_gaussian_process(
        x, y, torch.full((5,), 1e-10, dtype=DOUBLE), (0.3, 12.0),
        torch.Generator().manual_seed(0), restarts=3,
    )  # fmt: skip
    mean, sd = gp.posterior(x)
    assert torch.allclose(mean, y, atol=1e-3)
    assert float(sd.max()) < 1e-2
    assert 0.3 <= gp.hyperparameters.lengthscale <= 12.0
    # noise-only data pushes the lengthscale to a bound, never past it
    flat = fit_gaussian_process(
        x, torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0], dtype=DOUBLE),
        torch.full((5,), 1e-6, dtype=DOUBLE), (0.3, 12.0),
        torch.Generator().manual_seed(1), restarts=3,
    )  # fmt: skip
    assert 0.3 - 1e-9 <= flat.hyperparameters.lengthscale <= 12.0 + 1e-9


def test_cholesky_retry():
    x = torch.tensor([[0.5], [0.5], [1.0]], dtype=DOUBLE)  # duplicated input
    kernel = matern52_kernel(x, x, 1.0, 1.0)
    # singular: a negligible jitter fails every retry and raises
    with pytest.raises(torch.linalg.LinAlgError):
        robust_cholesky(kernel - 1e-12 * torch.eye(3), jitter=1e-20)
    # slightly indefinite: the first try fails, a doubled jitter works
    shifted = kernel - 1.5e-6 * torch.eye(3, dtype=DOUBLE)
    _, used = robust_cholesky(shifted, jitter=1e-6)
    assert used > 1e-6
    # the GP itself copes with duplicated inputs and no noise
    gp = condition_gaussian_process(
        x, torch.tensor([0.1, 0.1, 0.3], dtype=DOUBLE), torch.zeros(3),
        GaussianProcessHyperparameters(1.0, 1.0, 0.0),
    )  # fmt: skip
    assert torch.isfinite(gp.posterior(x)[0]).all()


# --------------------------------------------------------------------------- #
# 9-12. Bayesian optimisation
# --------------------------------------------------------------------------- #
def _value_key(value: float) -> int:
    return int(np.float64(value).view(np.int64)) & ((1 << 62) - 1)


def _stub_evaluate(functions, standard_error, seed, calls=None):
    """evaluate(candidates) for EIG_node(u) = functions[node](u) + noise of
    the given standard error, deterministic per (seed, node, value)."""

    def evaluate(candidates):
        if calls is not None:
            calls.append([c.target_node for c in candidates])
        out = []
        for c in candidates:
            rng = np.random.default_rng(
                [seed, c.target_node, _value_key(c.intervention_value)]
            )
            value = functions[c.target_node](c.intervention_value)
            value += standard_error * rng.standard_normal()
            out.append(
                InformationGainEstimate(
                    c, float(value), standard_error, torch.zeros(1)
                )
            )
        return out

    return evaluate


def _domain(node, c=3.0):
    return ValueDomain(node, -c, c, centre=0.0, scale=1.0)


def test_bayesian_optimisation_finds_the_maximum():
    peak = {0: lambda u: math.exp(-((u - 1.3) ** 2) / 0.5)}
    hits = 0
    for seed in range(10):
        optimiser = BayesianOptimisationValueOptimiser(
            initial_points=3,
            iterations=8,
            acquisition_grid_points=512,
            restarts=5,
        )
        result = optimiser.optimise(
            _stub_evaluate(peak, 0.02, seed), {0: _domain(0)},
            torch.Generator().manual_seed(seed),
        )[0]  # fmt: skip
        assert len(result.evaluations) == 3 + 8
        assert all(-3 <= e.standardised_value <= 3 for e in result.evaluations)
        assert [e.standardised_value for e in result.evaluations[:3]] == [
            -1.5,
            0.0,
            1.5,
        ]
        hits += abs(result.best_value - 1.3) < 0.2
    assert hits >= 9


def test_recommendation_follows_the_posterior_mean():
    smooth = {0.0: 2.0}  # u = 0: a large noisy value with a large error

    def evaluate(candidates):
        out = []
        for c in candidates:
            u = c.intervention_value
            if u in smooth:
                out.append(InformationGainEstimate(c, 2.0, 1.0, torch.zeros(1)))
            else:
                value = 0.6 - 0.1 * (u - 1.0) ** 2
                out.append(
                    InformationGainEstimate(c, value, 0.01, torch.zeros(1))
                )
        return out

    optimiser = BayesianOptimisationValueOptimiser(initial_points=5, iterations=0)
    result = optimiser.optimise(
        evaluate, {0: _domain(0)}, torch.Generator().manual_seed(0)
    )[0]
    raw_best = max(result.evaluations, key=lambda e: e.expected_information_gain)
    assert raw_best.standardised_value == 0.0
    assert result.best_standardised_value == pytest.approx(1.0)
    assert result.best_value_posterior_mean < 2.0


def test_lockstep_equals_separate_runs():
    functions = {
        1: lambda u: math.exp(-((u - 1.0) ** 2)),
        2: lambda u: 0.5 * math.exp(-((u + 2.0) ** 2) / 2),
        4: lambda u: 0.2 + 0.05 * u,
    }
    domains = {n: _domain(n) for n in functions}
    optimiser = BayesianOptimisationValueOptimiser(iterations=4, restarts=3)
    calls: list = []
    generator = torch.Generator().manual_seed(11)
    state = generator.get_state()
    together = optimiser.optimise(
        _stub_evaluate(functions, 0.02, 0, calls), domains, generator
    )
    assert len(calls) == 3 + 4
    assert all(sorted(c) == [1, 2, 4] for c in calls)
    for node in functions:
        alone_generator = torch.Generator()
        alone_generator.set_state(state)
        alone = optimiser.optimise(
            _stub_evaluate(functions, 0.02, 0), {node: domains[node]},
            alone_generator,
        )[node]  # fmt: skip
        mine = together[node]
        assert mine.best_value == alone.best_value
        assert [e.value for e in mine.evaluations] == [
            e.value for e in alone.evaluations
        ]
        assert np.array_equal(
            [e.lengthscale for e in mine.evaluations],
            [e.lengthscale for e in alone.evaluations],
            equal_nan=True,
        )
        assert mine.best_value_posterior_mean == alone.best_value_posterior_mean


def test_bi_level_optimisation_matters():
    width = 0.15
    functions = {
        0: lambda u: 0.3 + 0.6 * math.exp(-((u - 1.6) ** 2) / (2 * width**2)),
        1: lambda u: 0.5,
    }
    domains = {n: _domain(n) for n in functions}
    grid = GridValueOptimiser((-2, -1, 0, 1, 2))
    picks_a = 0
    for seed in range(5):
        evaluate = _stub_evaluate(functions, 0.01, seed)
        bo = BayesianOptimisationValueOptimiser(restarts=3).optimise(
            evaluate, domains, torch.Generator().manual_seed(seed)
        )
        chosen, _ = reevaluate_and_select(evaluate, bo)
        if chosen.target_node == 0 and abs(chosen.intervention_value - 1.6) < 0.2:
            picks_a += 1
        on_grid = grid.optimise(evaluate, domains, torch.Generator())
        chosen_grid, _ = reevaluate_and_select(evaluate, on_grid)
        assert chosen_grid.target_node == 1
    assert picks_a >= 4


# --------------------------------------------------------------------------- #
# 13. Outer re-evaluation
# --------------------------------------------------------------------------- #
def _inner(node, value, surrogate):
    domain = _domain(node)
    return ValueOptimisationResult(
        target_node=node, best_value=value, best_standardised_value=value,
        best_value_posterior_mean=surrogate, evaluations=[], domain=domain,
    )  # fmt: skip


def test_outer_level_selects_by_the_reevaluated_value():
    inner = {0: _inner(0, 1.0, 0.9), 1: _inner(1, -1.0, 0.2)}

    def evaluate_final(candidates):
        values = {0: 0.3, 1: 0.6}
        return [
            InformationGainEstimate(
                c, values[c.target_node], 0.01, torch.zeros(1)
            )
            for c in candidates
        ]

    chosen, scores = reevaluate_and_select(evaluate_final, inner)
    assert chosen.target_node == 1
    assert [s.inner_result for s in scores] == [inner[0], inner[1]]
    assert choice_is_clear(scores)


def test_choice_is_clear_flag():
    def score(value, se):
        return CandidateScore(
            CandidateExperiment(0, 0.0), value, se, float("nan"), None
        )

    assert not choice_is_clear([score(0.50, 0.02), score(0.47, 0.02)])
    assert choice_is_clear([score(0.50, 0.01), score(0.40, 0.01)])
    assert choice_is_clear([score(0.5, 0.1)])


def test_outer_level_uses_fresh_noise_and_final_fantasies(monkeypatch):
    stub = ConjugateStub()
    config = _cheap_config(final_num_fantasies=24)
    calls = []
    real = loop_module.estimate_expected_information_gain

    def recorder(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        loop_module, "estimate_expected_information_gain", recorder
    )
    seed = [5, 6]
    chosen, scores = select_next_experiment(
        stub, _empty_context(), 0, 2, PSI, config,
        ActiveLearningGenerators.from_seed(seed),
    )  # fmt: skip
    final = calls[-1]
    assert final["num_fantasies"] == 24 and final["compute_coherence_gap"]
    assert all(c["num_fantasies"] == config.num_fantasies for c in calls[:-1])
    expected = draw_fantasy_noise(
        [1, 3],
        24,
        stub,
        1,
        ActiveLearningGenerators.from_seed(seed).outer_fantasy,
    )
    for node in (1, 3):
        assert torch.equal(
            final["fantasy_noise"][node].residual_normal,
            expected[node].residual_normal,
        )
    inner_noise = calls[0]["fantasy_noise"][1].residual_normal
    assert not torch.equal(inner_noise[:8], expected[1].residual_normal[:8])
    best = max(scores, key=lambda s: s.expected_information_gain)
    assert chosen == best.candidate
    assert all(math.isfinite(s.coherence_gap) for s in scores)


# --------------------------------------------------------------------------- #
# 14-15. Ranking and exclusions
# --------------------------------------------------------------------------- #
def test_only_the_informative_node_is_chosen():
    stub = ConjugateStub()
    chosen, scores = select_next_experiment(
        stub, _empty_context(), 0, 2, PSI, _cheap_config(),
        ActiveLearningGenerators.from_seed(0),
    )  # fmt: skip
    assert chosen.target_node == 1
    assert {s.candidate.target_node for s in scores} == {1, 3}
    uninformative = next(s for s in scores if s.candidate.target_node == 3)
    assert uninformative.expected_information_gain == 0.0


def test_candidate_exclusions():
    assert candidate_target_nodes(5, 2, 4, True) == [0, 1, 3]
    assert candidate_target_nodes(5, 2, 4, False) == [0, 1, 2, 3]
    assert 4 not in candidate_target_nodes(5, 2, 4, False)


# --------------------------------------------------------------------------- #
# 16. Environment
# --------------------------------------------------------------------------- #
def test_environment_rows_have_the_exact_do_value():
    env = _stub_environment()
    rows = env.run_experiment(
        CandidateExperiment(1, 0.37, 5), np.random.default_rng(0)
    )
    assert rows.shape == (5, 4) and torch.all(rows[:, 1] == 0.37)
    mean, sd = env.true_query_distribution(0, 2, PSI)
    slope, variance = env.scm.do_dist(0, 2)
    assert torch.allclose(mean, slope * PSI) and torch.allclose(
        sd, torch.full_like(PSI, math.sqrt(variance))
    )


@pytest.mark.slow
def test_environment_moments_match_the_interventional_joint():
    env = _stub_environment()
    rows = env.run_experiment(
        CandidateExperiment(3, 1.5, 200_000), np.random.default_rng(1)
    ).numpy()
    slopes, covariance = env.scm.do_joint_dist(3)
    assert np.allclose(rows.mean(0), slopes * 1.5, atol=0.01)
    assert np.allclose(np.cov(rows.T), covariance, atol=0.01)


# --------------------------------------------------------------------------- #
# 17. Reproducibility
# --------------------------------------------------------------------------- #
def _stub_task():
    return ActiveLearningTask(0, _stub_environment(), 0, 2)


def test_reproducible_and_real_rows_independent_of_acquisition_settings():
    stub = ConjugateStub()
    strategies = make_strategies(
        _cheap_config(), ["expected_information_gain", "random"],
        ["bayesian_optimisation"],
    )  # fmt: skip
    first = run_active_learning_experiment(
        stub, [_stub_task()], strategies, [0, 1], 20
    ).steps
    second = run_active_learning_experiment(
        stub, [_stub_task()], strategies, [0, 1], 20
    ).steps
    assert first.equals(second)

    def random_rows(**settings):
        steps = run_active_learning(
            stub, _stub_environment(), _empty_context(), 0, 2, PSI,
            _cheap_config(acquisition="random", **settings),
            ActiveLearningGenerators.from_seed(3),
        )  # fmt: skip
        return [s.real_rows for s in steps]

    for a, b in zip(
        random_rows(),
        random_rows(bayesian_optimisation_iterations=5, num_fantasies=12),
    ):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# 18. End to end with a tiny joint model
# --------------------------------------------------------------------------- #
def test_end_to_end_tiny_joint_model():
    model = _tiny_joint_model()
    predictive = ModelApiPredictiveModel(model, 5, "cpu")
    B = np.zeros((5, 5))
    B[0, 1], B[1, 2], B[2, 4], B[0, 3], B[3, 4], B[0, 2] = (
        1,
        0.8,
        1.2,
        -0.7,
        0.5,
        0.4,
    )
    env = LinearSCMEnvironment(
        LinearSCM(B, np.full(5, 0.5)), {"kind": "normal", "scale": 1.5}
    )
    context = ExperimentContext(
        env.sample_observational(30, np.random.default_rng(0)),
        torch.zeros(30, 5, dtype=torch.bool),
    )
    for optimiser in ("bayesian_optimisation", "grid"):
        for acquisition in ACQUISITIONS:
            config = _cheap_config(
                acquisition=acquisition, value_optimiser=optimiser,
                num_fantasies=4, final_num_fantasies=4, quadrature_points=128,
                bayesian_optimisation_iterations=1, gaussian_process_restarts=1,
            )  # fmt: skip
            steps = run_active_learning(
                predictive, env, context, 2, 4, PSI, config,
                ActiveLearningGenerators.from_seed(0),
            )  # fmt: skip
            assert len(steps) == 2
            for step in steps:
                assert math.isfinite(step.kl_to_truth_after)
                assert math.isfinite(step.realised_information_gain)
                for s in step.scores:
                    assert math.isfinite(s.expected_information_gain)
    with pytest.raises(NotImplementedError):
        select_next_experiment(
            predictive, context, 2, 4, PSI,
            _cheap_config(include_observational_candidate=True, num_fantasies=4,
                          final_num_fantasies=4),
            ActiveLearningGenerators.from_seed(0),
        )  # fmt: skip


# --------------------------------------------------------------------------- #
# 19. Training support
# --------------------------------------------------------------------------- #
def test_training_support_warning_and_flag():
    support = TrainingSupport((10, 21), 11, 2)
    stub = ConjugateStub()
    with pytest.warns(UserWarning, match="training support"):
        steps = run_active_learning(
            stub, _stub_environment(), _empty_context(rows=20), 0, 2, PSI,
            _cheap_config(acquisition="random"),
            ActiveLearningGenerators.from_seed(0), training_support=support,
        )  # fmt: skip
    assert not steps[0].outside_training_support
    assert steps[1].outside_training_support
    assert "n_ctx_range" in steps[1].support_violations[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_active_learning(
            stub, _stub_environment(), _empty_context(rows=20), 0, 2, PSI,
            _cheap_config(acquisition="random"),
            ActiveLearningGenerators.from_seed(0),
            training_support=TrainingSupport((10, 100), 11, 2),
        )  # fmt: skip


def test_training_support_from_the_d5_config():
    from CITNP.datasets.create_mixed_regime_chunks import load_config

    config = Path(__file__).resolve().parents[1] / (
        "CITNP/experiments/configs/oracle_d5_iw_a15_sc.yaml"
    )
    support = TrainingSupport.from_data_config(load_config(str(config)), 5)
    assert support == TrainingSupport((64, 360), 33, 5)


# --------------------------------------------------------------------------- #
# 20. Config validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "settings",
    [
        {"acquisition": "greedy"},
        {"value_optimiser": "random_search"},
        {"kl_estimator": "exact"},
        {"fantasy_mode": "parallel"},
        {"value_domain": "anywhere"},
        {"experiment_budget": 0},
        {"num_fantasies": 1},
        {"rows_per_experiment": 0},
        {"inference_batch_size": 0},
        {"bayesian_optimisation_initial_points": 1},
        {"value_domain_half_width": 0.0},
        {"value_domain_half_width": -1.0},
    ],
)
def test_config_validation(settings):
    base = {"experiment_budget": 3}
    base.update(settings)
    with pytest.raises(ValueError):
        ActiveLearningConfig(**base)


def test_candidate_validation():
    with pytest.raises(ValueError):
        CandidateExperiment(1, None)
    with pytest.raises(ValueError):
        CandidateExperiment(None, 1.0)
    assert CandidateExperiment(None, None).is_observational


def test_active_learning_command(tmp_path):
    import yaml  # type: ignore[import-untyped]

    from CITNP.evaluation.__main__ import main
    from test_joint_prediction import RANK, _model_yaml

    model = _tiny_joint_model()
    ckpt = tmp_path / "joint.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    model_yaml = _model_yaml(
        tmp_path, prediction_head="joint", joint_covariance_rank=RANK
    )
    data = Path(__file__).resolve().parents[1] / (
        "CITNP/experiments/configs/oracle_d5_iw_a15_sc.yaml"
    )
    out = tmp_path / "al"
    main([
        "active-learning", "--ckpt", str(ckpt), "--model_config", str(model_yaml),
        "--data_config", str(data), "--environment", "example",
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
    assert (tmp_path / "al_surrogate_task0_seed0_step1.png").is_file()
    meta = yaml.safe_load((tmp_path / "al_meta.json").read_text())
    assert meta["tasks"][0]["treatment"] == 2
    import pandas as pd

    steps = pd.read_csv(tmp_path / "al_steps.csv")
    assert set(steps.strategy) == {
        "expected_information_gain/bayesian_optimisation",
        "expected_information_gain/grid", "random",
    }  # fmt: skip
