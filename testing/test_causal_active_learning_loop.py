"""The loop and the strategies (plan tests 9, 20, 21, 23; core)."""

import json
import math
import sys
import warnings
from pathlib import Path

import pytest
import torch

from causal_active_learning.acquisitions import NonIntervenableNodeError
from causal_active_learning.config import STRATEGIES, ActiveLearningConfig
from causal_active_learning.loop import (
    ActiveLearningGenerators,
    run_active_learning,
)
from causal_active_learning.support_check import TrainingSupport

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    ConjugateStub,
    StubEnvironment,
    cheap_config,
    empty_context,
    query,
)


def _run(config, D=4, seed=3, model=None, **kw):
    return run_active_learning(
        model or ConjugateStub(),
        StubEnvironment(D),
        empty_context(D=D),
        query(),
        config,
        ActiveLearningGenerators.from_seed(seed),
        **kw,
    )


# 20. Step 0 and rows per step
@pytest.mark.parametrize("R", [1, 3])
@pytest.mark.parametrize("D", [3, 5, 8])
def test_step_zero_and_rows_per_step(R, D):
    steps = _run(cheap_config(strategy="random", rows_per_acquisition=R), D=D)
    assert len(steps) == 2 + 1
    zero = steps[0]
    assert zero.step == 0 and zero.chosen is None and zero.real_rows is None
    assert zero.query_kl_to_truth is not None and zero.query_kl_to_truth > 0
    for step in steps[1:]:
        assert step.real_rows.shape == (R, D)
        assert step.num_variables == D
        if step.chosen.target_node is not None:
            assert step.newest_regime_size is not None
            assert step.newest_regime_size % R == 0


# 9. Only intervenable nodes, for every strategy
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategies_only_intervene_on_intervenable_nodes(strategy):
    nodes = (1, 3)
    config = cheap_config(
        strategy=strategy, intervenable_nodes=nodes, acquisition_budget=6
    )
    if strategy == "treatment":
        with pytest.raises(NonIntervenableNodeError):
            _run(config)
        return
    for seed in range(3):
        for step in _run(config, seed=seed)[1:]:
            node = step.chosen.target_node
            assert node is None or node in nodes
            assert step.intervenable_nodes == nodes


def test_random_baselines_use_the_observational_candidate():
    model = ConjugateStub(observational=False)
    config = cheap_config(
        strategy="random_fixed_value",
        acquisition_budget=12,
        fixed_intervention_value=0.25,
    )
    steps = _run(config, model=model)
    chosen = [s.chosen for s in steps[1:]]
    assert any(c.is_observational for c in chosen)
    assert all(c.intervention_value in (None, 0.25) for c in chosen)
    assert all(s.observational_candidate_available for s in steps)
    with pytest.warns(UserWarning, match="observational candidate"):
        eig = _run(cheap_config(), model=model)
    assert not any(s.observational_candidate_available for s in eig)


def test_random_strategy_values_stay_in_range():
    config = cheap_config(
        strategy="random",
        acquisition_budget=10,
        random_value_range=(-0.5, 0.25),
        include_observational_candidate=False,
    )
    for step in _run(config)[1:]:
        assert -0.5 <= step.chosen.intervention_value <= 0.25


# 21. Reproducibility (core part)
def test_reproducible_and_real_rows_independent_of_acquisition_settings():
    first, second = _run(cheap_config()), _run(cheap_config())
    for a, b in zip(first, second):
        assert a.chosen == b.chosen
        assert a.query_kl_to_truth == b.query_kl_to_truth
    random_a = _run(cheap_config(strategy="random"))
    random_b = _run(
        cheap_config(
            strategy="random",
            bayesian_optimisation_iterations=5,
            num_fantasies=12,
            fantasy_noise="shared",
        )
    )
    for a, b in zip(random_a[1:], random_b[1:]):
        assert torch.equal(a.real_rows, b.real_rows)


# Training support (core part of test 30) ----------------------------------------
def test_training_support_warning_flag_and_regime_sizes():
    support = TrainingSupport(
        trained_num_variables=(4,),
        context_size_range=(10, 21),
        minimum_observational_rows=11,
        maximum_interventional_regimes=2,
        do_value_range=(-6.0, 6.0),
    )
    config = cheap_config(
        strategy="random_fixed_value",
        fixed_intervention_value=0.1,
        include_observational_candidate=False,
    )
    with pytest.warns(UserWarning, match="training support"):
        steps = _run(config, training_support=support)
    assert not steps[0].outside_training_support
    assert steps[2].outside_training_support
    assert any("rows" in v for v in steps[2].support_violations)
    big_values = cheap_config(
        strategy="random_fixed_value",
        fixed_intervention_value=7.0,
        include_observational_candidate=False,
    )
    with pytest.warns(UserWarning, match="do-value"):
        steps = _run(
            big_values,
            training_support=TrainingSupport(do_value_range=(-6.0, 6.0)),
        )
    assert steps[1].outside_training_support
    with pytest.warns(UserWarning, match="not trained on"):
        _run(
            cheap_config(strategy="observational"),
            training_support=TrainingSupport(trained_num_variables=(5,)),
        )
    with pytest.raises(ValueError, match="architecture"):
        _run(
            cheap_config(),
            training_support=TrainingSupport(maximum_num_variables=3),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(
            cheap_config(strategy="observational"),
            training_support=TrainingSupport(trained_num_variables=(4,)),
        )


# 23. Config validation
@pytest.mark.parametrize(
    "settings",
    [
        {"strategy": "greedy"},
        {"value_search": "random_search"},
        {"kl_estimator": "exact"},
        {"value_domain": "anywhere"},
        {"fantasy_noise": "sometimes"},
        {"gaussian_process_noise": "free"},
        {"acquisition_budget": 0},
        {"num_fantasies": 7},
        {"final_num_fantasies": 4},
        {"rows_per_acquisition": 0},
        {"inference_batch_size": 0},
        {"observational_cost": 0.0},
        {"bayesian_optimisation_initial_points": 1},
        {"value_domain_half_width": 0.0},
        {"grid_num_values": 1},
        {"grid_standardised_values": (0.5,)},
        {"grid_standardised_values": (0.0, 1.5)},
        {"value_domain": "fixed_range"},
        {"value_domain_fixed_range": (-1.0, 1.0)},
        {"value_domain": "fixed_range", "value_domain_fixed_range": (1.0, -1.0)},
        {"random_value_range": (2.0, 2.0)},
        {"intervenable_nodes": (1, 1)},
    ],
)
def test_config_validation(settings):
    base = {"acquisition_budget": 3}
    base.update(settings)
    with pytest.raises(ValueError):
        ActiveLearningConfig(**base)


def test_config_defaults():
    config = ActiveLearningConfig(acquisition_budget=5)
    assert config.rows_per_acquisition == 3
    assert config.fantasy_noise == "independent"
    assert config.include_observational_candidate
    assert config.intervenable_nodes is None


# Entropy objective test 8: the KL objective reproduces the pre-change runs
def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def test_kl_objective_reproduces_the_runs_before_the_entropy_objectives():
    """Every strategy (and the EIG strategy with grid, Monte Carlo KL, shared
    noise, unequal costs, dense diagnostic), two steps: every column that
    existed before is identical, bit for bit, to the tables recorded by
    testing/make_causal_active_learning_kl_reference.py on the commit before
    the change."""
    from make_causal_active_learning_kl_reference import reference_tables

    path = Path(__file__).resolve().parent / "golden"
    reference = json.loads(
        (path / "causal_active_learning_kl_divergence.json").read_text()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = reference_tables(information_gain_objective="kl_divergence")
    for name, rows in reference.items():
        assert len(tables[name]) == len(rows), name
        for i, (new, old) in enumerate(zip(tables[name], rows)):
            assert set(old) <= set(new), (name, i, set(old) - set(new))
            different = {
                k: (old[k], new[k]) for k in old if not _same(new[k], old[k])
            }
            assert not different, (name, i, different)


# Entropy objective plan 3.5: realised counterparts
@pytest.mark.parametrize("objective", ["entropy_reduction", "kl_divergence"])
def test_realised_entropy_reduction_and_predicted_objectives(objective):
    config = cheap_config(
        intervenable_nodes=(1,),
        include_observational_candidate=False,
        information_gain_objective=objective,
    )
    steps = _run(config)
    assert steps[0].realised_entropy_reduction is None
    for step in steps:
        assert step.information_gain_objective == objective
    for step in steps[1:]:
        predicted = step.predicted_objective_values
        assert set(predicted) == {
            "entropy_reduction",
            "kl_divergence",
            "mixture_entropy_reduction",
        }
        assert predicted[objective] == step.predicted_information_gain
        # conjugate stub, intervention: H(after) does not depend on the
        # outcome, so the realised entropy reduction is the predicted one
        assert step.realised_entropy_reduction == pytest.approx(
            predicted["entropy_reduction"], abs=1e-9
        )
        assert step.realised_entropy_reduction > 0
    random = _run(cheap_config(strategy="random"))
    for step in random[1:]:
        assert math.isfinite(step.realised_entropy_reduction)
        assert step.predicted_objective_values == {}
