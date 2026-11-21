"""CITNP integration of causal_active_learning: the model adapter, model_api
additions, the linear SCM environment, the support reader and the command."""

import math
import warnings
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_active_learning.acquisitions import Acquisition, InterventionalQuery
from causal_active_learning.config import STRATEGIES
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.integrations.citnp.environment import (
    LinearSCMEnvironment,
)
from causal_active_learning.integrations.citnp.model import (
    CITNPPredictiveModel,
)
from causal_active_learning.integrations.citnp.experiment import (
    build_tasks,
    load_experiment,
    make_strategies,
    run_experiment,
)
from causal_active_learning.integrations.citnp.support import (
    training_support_from_data_config,
)
from causal_active_learning.loop import (
    ActiveLearningGenerators,
    run_active_learning,
)
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
from causal_active_learning_stubs import (
    PSI,
    cheap_config,
    empty_context,
)
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
FIGURE_4 = REPO / "causal_active_learning/integrations/citnp/configs"
FIGURE_4_FILES = sorted(FIGURE_4.glob("*.yaml"))


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
            augmented.values[None],
            augmented.intervened[None],
            torch.tensor([0]),
            torch.tensor([1]),
            torch.zeros(1, 2),
            normalise=True,
        )
        scratch, mean_s, std_s = _model_inputs(
            augmented.values.numpy(),
            augmented.intervened.numpy(),
            0,
            1,
            np.zeros(2),
            zero_flags=False,
            normalise=True,
        )
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
    adapter = CITNPPredictiveModel(model, "cpu")
    context = empty_context(rows=30, D=5)
    g = torch.Generator().manual_seed(1)
    candidates = [Acquisition(1, 0.7), Acquisition(3, -1.2)]
    noise = [adapter.draw_fantasy_noise(5, 6, 1, g) for _ in candidates]
    rows = adapter.sample_fantasies([context] * 2, candidates, noise)
    assert rows.shape == (2, 6, 1, 5)
    for c, cand in enumerate(candidates):
        assert torch.all(
            rows[c, :, 0, cand.target_node] == cand.intervention_value
        )
        joint = predict_joint_mixture(
            model,
            context.values.numpy(),
            context.intervened.numpy(),
            cand.target_node,
            np.array([cand.intervention_value]),
            device="cpu",
        )
        reference = sample_joint_rows(
            joint,
            np.array([cand.intervention_value]),
            6,
            noise=noise[c].row(0),
        )[0, 0]
        assert torch.allclose(rows[c, :, 0], reference, atol=1e-4)


@pytest.mark.parametrize("sampling", ["autoregressive", "independent"])
def test_multi_row_fantasies(sampling):
    adapter = CITNPPredictiveModel(
        tiny_joint_model(), "cpu", fantasy_row_sampling=sampling
    )
    g = torch.Generator().manual_seed(0)
    R = 3
    candidates = [Acquisition(1, 1.0, R), Acquisition(3, 0.5, R)]
    noise = [adapter.draw_fantasy_noise(5, 4, R, g) for _ in candidates]
    rows = adapter.sample_fantasies([empty_context(D=5)] * 2, candidates, noise)
    assert rows.shape == (2, 4, R, 5)
    assert torch.all(rows[0, :, :, 1] == 1.0) and torch.all(
        rows[1, :, :, 3] == 0.5
    )
    assert torch.isfinite(rows).all()


def test_observational_fantasies_are_not_available():
    adapter = CITNPPredictiveModel(tiny_joint_model(), "cpu")
    assert adapter.supports_observational_prediction is False
    noise = [adapter.draw_fantasy_noise(5, 4, 1, torch.Generator())]
    with pytest.raises(NotImplementedError):
        adapter.sample_fantasies(
            [empty_context(D=5)], [Acquisition(None, None)], noise
        )
    with pytest.raises(ValueError):
        from CITNP.models.mixed_context_model import INTERVENTION_ENCODINGS

        CITNPPredictiveModel(
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
    rows = sample_joint_rows(
        jp,
        np.array([0.0]),
        16,
        noise=JointSampleNoise(
            component_uniform=torch.full((16,), 1e-9, dtype=DOUBLE),
            factor_normal=torch.zeros(16, 2, dtype=DOUBLE),
            residual_normal=torch.zeros(16, 4, dtype=DOUBLE),
        ),
    )
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
            joint,
            x[b].numpy(),
            m[b].numpy(),
            int(T[b]),
            t[b].numpy(),
            device="cpu",
        )
        for name in (
            "component_mean",
            "covariance_factor",
            "covariance_diagonal",
            "log_weights",
        ):
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
    rows = env.run(Acquisition(1, 0.37, 5), np.random.default_rng(0))
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
    rows = env.run(Acquisition(3, 1.5, 200_000), np.random.default_rng(1)).numpy()
    slopes, covariance = env.scm.do_joint_dist(3)
    assert np.allclose(rows.mean(0), slopes * 1.5, atol=0.01)
    assert np.allclose(np.cov(rows.T), covariance, atol=0.01)


# --------------------------------------------------------------------------- #
# Autoregressive rows through model_api (plan test 5 at the adapter)
# --------------------------------------------------------------------------- #
def test_adapter_rows_condition_on_earlier_rows_of_the_same_fantasy(monkeypatch):
    from causal_active_learning.integrations.citnp import model as adapter_module

    seen = []
    real = adapter_module.predict_joint_mixture_batched

    def recorder(model, ctx_x, ctx_int, T, t, **kwargs):
        seen.append(ctx_x.clone())
        return real(model, ctx_x, ctx_int, T, t, **kwargs)

    monkeypatch.setattr(adapter_module, "predict_joint_mixture_batched", recorder)
    adapter = CITNPPredictiveModel(tiny_joint_model(), "cpu")
    context = empty_context(rows=12, D=5)
    noise = [
        adapter.draw_fantasy_noise(5, 4, 3, torch.Generator().manual_seed(0))
    ]
    rows = adapter.sample_fantasies(
        [context], [Acquisition.intervention(1, 0.4, 3)], noise
    )[0]  # [S, R, D]
    assert [x.shape[1] for x in seen] == [12, 13, 14]
    for s in range(4):
        assert torch.equal(seen[1][s, 12], rows[s, 0])
        assert torch.equal(seen[2][s, 12:14], rows[s, :2])
    assert torch.all(rows[:, :, 1] == 0.4)


def test_per_query_do_values():
    jp = predict_joint_mixture_batched(
        tiny_joint_model(),
        torch.randn(2, 20, 5, dtype=DOUBLE),
        torch.zeros(2, 20, 5, dtype=torch.bool),
        torch.tensor([1, 3]),
        torch.tensor([[0.5], [-2.0]], dtype=DOUBLE),
        device="cpu",
    )
    rows = sample_joint_rows(
        jp,
        torch.tensor([[0.5], [-2.0]], dtype=DOUBLE),
        6,
        torch.Generator().manual_seed(0),
    )
    assert torch.all(rows[0, :, :, 1] == 0.5) and torch.all(
        rows[1, :, :, 3] == -2.0
    )


# --------------------------------------------------------------------------- #
# End to end (plan test 29)
# --------------------------------------------------------------------------- #
def _initial(env, rows=30):
    values = env.sample_observational(rows, np.random.default_rng(0))
    return ExperimentalContext(
        values, torch.zeros(values.shape, dtype=torch.bool)
    )


def test_end_to_end_every_strategy_and_value_search():
    adapter = CITNPPredictiveModel(tiny_joint_model(), "cpu")
    env = example_environment()
    query = InterventionalQuery(2, 4, PSI)
    for search in ("bayesian_optimisation", "grid"):
        for strategy in STRATEGIES:
            config = cheap_config(
                strategy=strategy,
                value_search=search,
                rows_per_acquisition=3,
                num_fantasies=8,
                final_num_fantasies=8,
                quadrature_points=128,
                bayesian_optimisation_iterations=1,
                gaussian_process_restarts=1,
                grid_num_values=3,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                steps = run_active_learning(
                    adapter,
                    env,
                    _initial(env),
                    query,
                    config,
                    ActiveLearningGenerators.from_seed(0),
                )
            assert len(steps) == 3
            for step in steps:
                assert math.isfinite(step.query_kl_to_truth)
                for s in step.scores:
                    if not s.acquisition.is_observational:
                        assert math.isfinite(s.expected_information_gain)
            if strategy == "expected_information_gain":
                assert not steps[1].observational_candidate_available


@pytest.mark.parametrize("D", [2, 3, 5])
@pytest.mark.parametrize("R", [1, 3])
@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_end_to_end_dimensions_rows_and_noise_modes(D, R, mode):
    adapter = CITNPPredictiveModel(tiny_joint_model(), "cpu")
    B = np.zeros((D, D))
    B[0, D - 1] = 0.8
    env = LinearSCMEnvironment(
        LinearSCM(B, np.full(D, 0.5)), {"kind": "normal", "scale": 1.5}
    )
    config = cheap_config(
        rows_per_acquisition=R,
        fantasy_noise=mode,
        num_fantasies=8,
        final_num_fantasies=8,
        quadrature_points=128,
        bayesian_optimisation_iterations=1,
        gaussian_process_restarts=1,
    )
    with pytest.warns(UserWarning, match="observational candidate"):
        steps = run_active_learning(
            adapter,
            env,
            _initial(env),
            InterventionalQuery(0, D - 1, PSI),
            config,
            ActiveLearningGenerators.from_seed(1),
        )
    assert [s.real_rows.shape for s in steps[1:]] == [(R, D)] * 2
    for step in steps:
        assert math.isfinite(step.query_kl_to_truth)
        assert all(
            math.isfinite(s.expected_information_gain) for s in step.scores
        )


# --------------------------------------------------------------------------- #
# Training support (plan test 30)
# --------------------------------------------------------------------------- #
def test_training_support_from_the_d5_config():
    from CITNP.datasets.create_mixed_regime_chunks import load_config

    support = training_support_from_data_config(load_config(str(D5_CONFIG)))
    assert support.trained_num_variables == (5,)
    assert support.context_size_range == (64, 360)
    assert support.minimum_observational_rows == 33
    assert support.maximum_interventional_regimes == 5
    assert support.do_value_range == (-6.0, 6.0)
    assert support.maximum_num_variables is None
    assert not support.observational_queries_trained


def test_training_support_checks_with_citnp():
    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.models.mixed_context_model import (
        JointExtraNodeMixedContextMoGModel,
    )

    cfg = load_config(str(D5_CONFIG))
    extra_node = build(JointExtraNodeMixedContextMoGModel, num_nodes=3)
    support = training_support_from_data_config(cfg, extra_node)
    assert support.maximum_num_variables == 3
    env = example_environment()
    adapter = CITNPPredictiveModel(tiny_joint_model(), "cpu")
    with pytest.raises(ValueError, match="architecture"):
        run_active_learning(
            adapter,
            env,
            _initial(env),
            InterventionalQuery(2, 4, PSI),
            cheap_config(strategy="observational"),
            ActiveLearningGenerators.from_seed(0),
            training_support=support,
        )
    support = training_support_from_data_config(cfg, tiny_joint_model())
    big = cheap_config(
        strategy="random_fixed_value",
        fixed_intervention_value=7.0,
        include_observational_candidate=False,
        rows_per_acquisition=3,
    )
    with pytest.warns(UserWarning) as record:
        steps = run_active_learning(
            adapter,
            env,
            _initial(env, rows=30),
            InterventionalQuery(2, 4, PSI),
            big,
            ActiveLearningGenerators.from_seed(0),
            training_support=support,
        )
    messages = " ".join(str(w.message) for w in record)
    assert "below the training minimum 64" in messages
    assert "do-value" in messages
    assert all(s.outside_training_support for s in steps)
    assert steps[1].newest_regime_size == 3
    B = np.zeros((4, 4))
    B[0, 3] = 0.5
    env4 = LinearSCMEnvironment(
        LinearSCM(B, np.full(4, 0.5)), {"kind": "normal", "scale": 1.0}
    )
    with pytest.warns(UserWarning, match="not trained on"):
        run_active_learning(
            adapter,
            env4,
            _initial(env4, rows=70),
            InterventionalQuery(0, 3, PSI),
            cheap_config(strategy="observational"),
            ActiveLearningGenerators.from_seed(0),
            training_support=support,
        )


# --------------------------------------------------------------------------- #
# Experiment configs, pairing and the commands (plan tests 21, 23, 31)
# --------------------------------------------------------------------------- #
def _data_config():
    from CITNP.datasets.create_mixed_regime_chunks import load_config

    return load_config(str(D5_CONFIG))


@pytest.mark.parametrize("path", FIGURE_4_FILES, ids=lambda p: p.stem)
def test_figure_4_configs(path):
    spec = load_experiment(str(path))
    assert spec.edges == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)]
    assert (spec.treatment, spec.outcome) == (2, 4)
    assert spec.treatment_range == (2.0, 5.0) and spec.num_query_samples == 16
    assert (spec.num_tasks, spec.restarts_per_task) == (10, 3)
    config = spec.config("random")
    assert config.acquisition_budget == 50 and config.rows_per_acquisition == 3
    assert config.fixed_intervention_value == 0.0
    in_distribution = "in_distribution" in path.stem
    assert spec.initial_observational_rows == (64 if in_distribution else 5)
    limit = 6.0 if in_distribution else 7.0
    assert config.random_value_range == (-limit, limit)
    assert config.value_domain_fixed_range == (-limit, limit)
    expected = None if path.stem.endswith("all_intervenable") else (0, 1, 3, 4)
    assert config.intervenable_nodes == expected
    tasks = build_tasks(spec, _data_config(), 0, [0])
    assert tasks[0].environment.graph.sum() == 7
    assert len(tasks[0].query.treatment_values) == 16
    assert torch.all(
        (tasks[0].query.treatment_values >= 2)
        & (tasks[0].query.treatment_values <= 5)
    )


def test_experiment_validation(tmp_path):
    import yaml  # type: ignore[import-untyped]

    from causal_active_learning.integrations.citnp.experiment import (
        experiment_from_mapping,
    )

    base = yaml.safe_load(
        (FIGURE_4 / "abci_figure_4_all_intervenable.yaml").read_text()
    )
    experiment_from_mapping(base)
    bad = [
        {**base, "colour": "red"},
        {**base, "query": {**base["query"], "treatment_values": [1.0]}},
        {
            **base,
            "task_source": {**base["task_source"], "edges": [[0, 1], [1, 0]]},
        },
        {**base, "task_source": {**base["task_source"], "edges": [[0, 9]]}},
        {**base, "strategies": ["greedy"]},
        {**base, "active_learning": {**base["active_learning"], "budget": 3}},
        {
            **base,
            "active_learning": {
                **base["active_learning"],
                "value_domain": "fixed_range",
                "value_domain_fixed_range": None,
            },
        },
    ]
    for mapping in bad:
        with pytest.raises(ValueError):
            experiment_from_mapping(mapping)


def test_pairing_across_settings_strategies_and_restarts():
    cfg = _data_config()
    a = load_experiment(str(FIGURE_4 / "abci_figure_4_all_intervenable.yaml"))
    b = load_experiment(
        str(FIGURE_4 / "abci_figure_4_treatment_not_intervenable.yaml")
    )
    tasks_a = build_tasks(a, cfg, 5, [0, 1])
    tasks_b = build_tasks(b, cfg, 5, [0, 1])
    for ta, tb in zip(tasks_a, tasks_b):
        assert np.array_equal(ta.environment.scm.B, tb.environment.scm.B)
        assert np.array_equal(ta.environment.scm.s2, tb.environment.scm.s2)
        assert torch.equal(ta.query.treatment_values, tb.query.treatment_values)
    assert not np.array_equal(
        tasks_a[0].environment.scm.B, tasks_a[1].environment.scm.B
    )
    adapter = CITNPPredictiveModel(
        tiny_joint_model(), "cpu", minimum_observational_rows=5
    )
    strategies = make_strategies(
        lambda name: a.config(
            name,
            acquisition_budget=1,
            num_fantasies=8,
            final_num_fantasies=8,
            quadrature_points=64,
        ),
        ["random", "observational"],
    )
    tables = run_experiment(
        adapter, "a", tasks_a[:1], [0, 1], strategies, 12, seed=5
    )
    steps = tables.steps
    zero = (
        steps[steps.step == 0]
        .set_index(["restart", "strategy"])
        .query_kl_to_truth
    )
    assert zero[(0, "random")] == zero[(0, "observational")]
    assert zero[(0, "random")] != zero[(1, "random")]


def _write_tiny_checkpoint(tmp_path):
    ckpt = tmp_path / "joint.pt"
    torch.save({"model_state_dict": tiny_joint_model().state_dict()}, ckpt)
    model_yaml = _model_yaml(
        tmp_path, prediction_head="joint", joint_covariance_rank=RANK
    )
    return str(ckpt), str(model_yaml)


SMOKE = [
    "--acquisition_budget",
    "2",
    "--num_fantasies",
    "8",
    "--final_num_fantasies",
    "8",
    "--quadrature_points",
    "64",
    "--bayesian_optimisation_iterations",
    "1",
    "--gaussian_process_restarts",
    "1",
    "--task_indices",
    "0",
    "--restart_indices",
    "0",
    "--device",
    "cpu",
]


def test_figure_4_smoke_runs_merge_and_panels(tmp_path):
    from causal_active_learning.integrations.citnp.command import main

    ckpt, model_yaml = _write_tiny_checkpoint(tmp_path)
    prefixes = []
    for path in FIGURE_4_FILES:
        out = str(tmp_path / path.stem)
        extra = (
            []
            if "in_distribution" in path.stem
            else ["--minimum_observational_rows", "5"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(
                [
                    "run",
                    "--ckpt",
                    ckpt,
                    "--model_config",
                    model_yaml,
                    "--data_config",
                    str(D5_CONFIG),
                    "--experiment",
                    str(path),
                    "--out",
                    out,
                    *SMOKE,
                    *extra,
                ]
            )
        for name in (
            "steps",
            "scores",
            "bayesian_optimisation_trace",
            "surrogate",
            "summary",
            "meta",
        ):
            suffix = ".json" if name == "meta" else ".csv"
            assert Path(f"{out}_{name}{suffix}").is_file(), (path.stem, name)
        prefixes.append(out)
    import pandas as pd

    steps = pd.read_csv(prefixes[0] + "_steps.csv")
    assert set(steps.step) == {0, 1, 2}
    assert set(steps.strategy) == {
        "expected_information_gain",
        "observational",
        "random_fixed_value",
        "random",
        "exhaustive_true_gain",
    }
    literal = [p for p in prefixes if "in_distribution" not in p]
    assert pd.read_csv(literal[0] + "_steps.csv").outside_training_support.all()
    main(["merge", "--inputs", *literal, "--out", str(tmp_path / "merged")])
    assert (tmp_path / "merged_summary.csv").is_file()
    figure = tmp_path / "panels.png"
    main(
        [
            "plot-panels",
            "--summaries",
            literal[0] + "_summary.csv",
            literal[1] + "_summary.csv",
            "--experiments",
            str(FIGURE_4 / "abci_figure_4_all_intervenable.yaml"),
            str(FIGURE_4 / "abci_figure_4_treatment_not_intervenable.yaml"),
            "--titles",
            "All nodes intervenable",
            "X3 not intervenable",
            "--out",
            str(figure),
        ]
    )
    assert figure.is_file()


def test_literal_start_needs_the_lower_row_minimum(tmp_path):
    from causal_active_learning.integrations.citnp.command import main

    ckpt, model_yaml = _write_tiny_checkpoint(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--ckpt",
                ckpt,
                "--model_config",
                model_yaml,
                "--data_config",
                str(D5_CONFIG),
                "--experiment",
                str(FIGURE_4 / "abci_figure_4_all_intervenable.yaml"),
                "--out",
                str(tmp_path / "x"),
                *SMOKE,
            ]
        )
    with pytest.raises(SystemExit):  # an experiment file and ad hoc arguments
        main(
            [
                "run",
                "--ckpt",
                ckpt,
                "--model_config",
                model_yaml,
                "--data_config",
                str(D5_CONFIG),
                "--experiment",
                str(FIGURE_4 / "abci_figure_4_all_intervenable.yaml"),
                "--environment",
                "prior",
                "--out",
                str(tmp_path / "x"),
            ]
        )


def test_ad_hoc_example_run(tmp_path):
    import json

    from causal_active_learning.integrations.citnp.command import main

    ckpt, model_yaml = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "al"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(
            [
                "run",
                "--ckpt",
                ckpt,
                "--model_config",
                model_yaml,
                "--data_config",
                str(D5_CONFIG),
                "--environment",
                "example",
                "--fixed_graph_edges",
                "0->1,1->2,2->4,0->3,3->4,0->2",
                "--treatment",
                "2",
                "--outcome",
                "4",
                "--strategies",
                "expected_information_gain,random",
                "--value_search",
                "bayesian_optimisation,grid",
                "--seeds",
                "0",
                "--dense_value_diagnostic",
                "--dense_value_diagnostic_points",
                "3",
                "--acquisition_budget",
                "1",
                "--num_fantasies",
                "8",
                "--final_num_fantasies",
                "8",
                "--quadrature_points",
                "64",
                "--bayesian_optimisation_iterations",
                "1",
                "--gaussian_process_restarts",
                "1",
                "--rows_per_acquisition",
                "1",
                "--device",
                "cpu",
                "--out",
                str(out),
            ]
        )
    import pandas as pd

    steps = pd.read_csv(f"{out}_steps.csv")
    assert set(steps.strategy) == {
        "expected_information_gain/bayesian_optimisation",
        "expected_information_gain/grid",
        "random",
    }
    dense = pd.read_csv(f"{out}_dense_value_diagnostic.csv")
    assert set(dense.noise_mode) == {"independent", "shared"}
    assert dense.distance_to_recommendation.notna().any()
    meta = json.loads(Path(f"{out}_meta.json").read_text())
    assert meta["tasks"][0]["treatment"] == 2
