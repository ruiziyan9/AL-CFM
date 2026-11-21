"""GP surrogate, noise policy, BO and grid value search (plan tests 11-14,
16-18; core, stub evaluate functions)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_active_learning.information_gain import InformationGainEstimate
from causal_active_learning.selection import reevaluate_and_select
from causal_active_learning.value_search import make_value_search
from causal_active_learning.value_search.bayesian_optimisation import (
    BayesianOptimisationValueSearch,
)
from causal_active_learning.value_search.gaussian_process import (
    NUGGET_BOUNDS,
    GaussianProcessHyperparameters,
    condition_gaussian_process,
    fit_gaussian_process,
    matern52_kernel,
    robust_cholesky,
)
from causal_active_learning.value_search.grid import GridValueSearch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    cheap_config,
    domain,
    stub_evaluate,
)

DOUBLE = torch.float64


# 11. GP
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

    K = (
        k(x, x)
        + torch.diag(noise / scale**2 + h.nugget)
        + jitter * torch.eye(len(x))
    )
    cross = k(x, test)
    mean = h.constant_mean + cross.T @ torch.linalg.solve(K, ys - h.constant_mean)
    var = h.output_scale - (cross * torch.linalg.solve(K, cross)).sum(0)
    return mean_y + scale * mean, scale * var.clamp(min=0).sqrt()


@pytest.mark.parametrize("nugget", [0.0, 0.003])
def test_gaussian_process_against_dense_reference(nugget):
    x = torch.tensor([-0.9, -0.3, 0.1, 0.4, 0.8], dtype=DOUBLE)
    y = torch.tensor([0.1, 0.4, 0.9, 0.7, 0.2], dtype=DOUBLE)
    noise = torch.tensor([1e-3, 4e-3, 2e-3, 1e-2, 5e-3], dtype=DOUBLE)
    h = GaussianProcessHyperparameters(0.4, 1.3, 0.2, nugget)
    gp = condition_gaussian_process(x[:, None], y, noise, h)
    test = torch.linspace(-1, 1, 41, dtype=DOUBLE)
    mean, sd = gp.posterior(test[:, None])
    ref_mean, ref_sd = _reference_posterior(x, y, noise, h, test)
    assert torch.allclose(mean, ref_mean, atol=1e-8)
    assert torch.allclose(sd, ref_sd, atol=1e-8)


def test_gaussian_process_interpolates_and_respects_bounds():
    x = torch.tensor([[-0.8], [-0.3], [0.1], [0.4], [0.9]], dtype=DOUBLE)
    y = torch.sin(3 * x[:, 0])
    gp = fit_gaussian_process(
        x,
        y,
        torch.full((5,), 1e-10, dtype=DOUBLE),
        (0.1, 4.0),
        torch.Generator().manual_seed(0),
        restarts=3,
    )
    mean, sd = gp.posterior(x)
    assert torch.allclose(mean, y, atol=2e-2)
    assert 0.1 <= gp.hyperparameters.lengthscale <= 4.0
    low, high = NUGGET_BOUNDS
    assert low - 1e-12 <= gp.hyperparameters.nugget <= high + 1e-12


def test_cholesky_retry():
    x = torch.tensor([[0.5], [0.5], [1.0]], dtype=DOUBLE)  # duplicated input
    kernel = matern52_kernel(x, x, 1.0, 1.0)
    with pytest.raises(torch.linalg.LinAlgError):
        robust_cholesky(kernel - 1e-12 * torch.eye(3), jitter=1e-20)
    _, used = robust_cholesky(kernel - 1.5e-6 * torch.eye(3, dtype=DOUBLE), 1e-6)
    assert used > 1e-6
    gp = condition_gaussian_process(
        x,
        torch.tensor([0.1, 0.1, 0.3], dtype=DOUBLE),
        torch.zeros(3),
        GaussianProcessHyperparameters(1.0, 1.0, 0.0),
    )
    assert torch.isfinite(gp.posterior(x)[0]).all()


# 12. Noise policy
def test_noise_policy_resolution_and_nugget_bounds():
    assert cheap_config().resolved_gaussian_process_noise() == "standard_error"
    shared = cheap_config(fantasy_noise="shared")
    assert shared.resolved_gaussian_process_noise() == "fitted"
    explicit = cheap_config(
        fantasy_noise="shared", gaussian_process_noise="fixed"
    )
    assert explicit.resolved_gaussian_process_noise() == "fixed"
    x = torch.linspace(-1, 1, 7, dtype=DOUBLE)[:, None]
    y = torch.sin(2 * x[:, 0]) + 0.05 * torch.randn(
        7, generator=torch.Generator().manual_seed(0), dtype=DOUBLE
    )
    se2 = torch.full((7,), 0.03**2, dtype=DOUBLE)
    g = torch.Generator().manual_seed(1)
    standard = fit_gaussian_process(
        x, y, se2, (0.1, 4.0), g, noise_policy="standard_error"
    )
    assert (
        NUGGET_BOUNDS[0] - 1e-12
        <= standard.hyperparameters.nugget
        <= NUGGET_BOUNDS[1] + 1e-12
    )
    fitted = fit_gaussian_process(x, y, se2, (0.1, 4.0), g, noise_policy="fitted")
    upper = float(se2.mean() / y.var(unbiased=False))
    assert (
        1e-6 - 1e-12 <= fitted.hyperparameters.nugget <= max(upper, 2e-6) + 1e-12
    )
    assert torch.all(fitted.noise_variances == 0)
    fixed = fit_gaussian_process(x, y, se2, (0.1, 4.0), g, noise_policy="fixed")
    assert fixed.hyperparameters.nugget == 0.0
    for gp in (standard, fitted, fixed):
        assert torch.isfinite(gp.posterior(x)[1]).all()


# 13. BO finds a known maximum
@pytest.mark.parametrize("centre, half_width", [(0.0, 1.0), (2.0, 3.0)])
def test_bayesian_optimisation_finds_the_maximum(centre, half_width):
    def peak(v):
        u = (v - centre) / half_width
        return math.exp(-((u - 0.43) ** 2) / 0.06)

    hits = 0
    for seed in range(10):
        result = BayesianOptimisationValueSearch().search(
            stub_evaluate({0: peak}, 0.02, seed),
            {0: domain(0, centre, half_width)},
            torch.Generator().manual_seed(seed),
        )[0]
        assert len(result.evaluations) == 3 + 8
        assert all(-1 <= e.standardised_value <= 1 for e in result.evaluations)
        assert [e.standardised_value for e in result.evaluations[:3]] == [
            -0.5,
            0.0,
            0.5,
        ]
        assert result.best_value == pytest.approx(
            centre + half_width * result.best_standardised_value
        )
        hits += abs(result.best_standardised_value - 0.43) < 0.07
    assert hits >= 9


# 14. Recommendation rule
def test_recommendation_follows_the_posterior_mean():
    def evaluate(acquisitions):
        out = []
        for a in acquisitions:
            u = a.intervention_value
            if abs(u) < 1e-12:  # one large noisy value with a large error
                out.append(InformationGainEstimate(a, 2.0, 1.0, torch.zeros(1)))
            else:
                value = 0.6 - 0.3 * (u - 0.33) ** 2
                out.append(
                    InformationGainEstimate(a, value, 0.01, torch.zeros(1))
                )
        return out

    search = BayesianOptimisationValueSearch(initial_points=5, iterations=0)
    result = search.search(
        evaluate, {0: domain(0)}, torch.Generator().manual_seed(0)
    )[0]
    raw_best = max(result.evaluations, key=lambda e: e.expected_information_gain)
    assert raw_best.standardised_value == pytest.approx(0.0)
    assert result.best_standardised_value != pytest.approx(0.0)


# 16. Lockstep equals separate searches
def test_lockstep_equals_separate_runs():
    functions = {
        1: lambda u: math.exp(-((u - 0.3) ** 2) / 0.1),
        2: lambda u: 0.5 * math.exp(-((u + 0.6) ** 2) / 0.2),
        4: lambda u: 0.2 + 0.05 * u,
    }
    domains = {n: domain(n) for n in functions}
    search = BayesianOptimisationValueSearch(iterations=4, restarts=3)
    calls: list = []
    generator = torch.Generator().manual_seed(11)
    state = generator.get_state()
    together = search.search(
        stub_evaluate(functions, 0.02, 0, calls), domains, generator
    )
    assert len(calls) == 3 + 4 and all(sorted(c) == [1, 2, 4] for c in calls)
    for node in functions:
        alone_generator = torch.Generator()
        alone_generator.set_state(state)
        alone = search.search(
            stub_evaluate(functions, 0.02, 0),
            {node: domains[node]},
            alone_generator,
        )[node]
        mine = together[node]
        assert mine.best_value == alone.best_value
        assert [e.value for e in mine.evaluations] == [
            e.value for e in alone.evaluations
        ]
        assert np.array_equal(
            [e.nugget for e in mine.evaluations],
            [e.nugget for e in alone.evaluations],
            equal_nan=True,
        )


# 17. Bi-level matters
def test_bi_level_optimisation_matters():
    width = 0.05
    functions = {
        0: lambda u: 0.3 + 0.6 * math.exp(-((u - 0.53) ** 2) / (2 * width**2)),
        1: lambda u: 0.5,
    }
    domains = {n: domain(n) for n in functions}
    grid_values = (-2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3)
    assert all(functions[0](u) < 0.32 for u in grid_values)
    grid = GridValueSearch(grid_values)
    picks_a = 0
    for seed in range(5):
        evaluate = stub_evaluate(functions, 0.01, seed)
        bo = BayesianOptimisationValueSearch(restarts=3).search(
            evaluate, domains, torch.Generator().manual_seed(seed)
        )
        chosen, _ = reevaluate_and_select(evaluate, bo)
        if (
            chosen.target_node == 0
            and abs(chosen.intervention_value - 0.53) < 0.07
        ):
            picks_a += 1
        chosen_grid, _ = reevaluate_and_select(
            evaluate, grid.search(evaluate, domains, torch.Generator())
        )
        assert chosen_grid.target_node == 1
    assert picks_a >= 4


# 18. Grid search
def test_grid_search_values():
    search = make_value_search(
        cheap_config(value_search="grid", grid_num_values=5)
    )
    assert np.allclose(search.standardised_values, [-1, -0.5, 0, 0.5, 1])
    explicit = make_value_search(
        cheap_config(
            value_search="grid",
            grid_num_values=5,
            grid_standardised_values=(-0.2, 0.7),
        )
    )
    assert explicit.standardised_values == (-0.2, 0.7)
    calls: list = []
    result = explicit.search(
        stub_evaluate({3: lambda u: u}, 0.0, 0, calls),
        {3: domain(3)},
        torch.Generator(),
    )[3]
    assert len(calls) == 1 and result.best_standardised_value == 0.7
