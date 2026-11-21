"""GP surrogate, BO and grid value search (core, stub evaluate functions)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_active_learning.information_gain import InformationGainEstimate
from causal_active_learning.selection import reevaluate_and_select
from causal_active_learning.value_search.bayesian_optimisation import (
    BayesianOptimisationValueOptimiser,
)
from causal_active_learning.value_search.gaussian_process import (
    GaussianProcessHyperparameters,
    condition_gaussian_process,
    fit_gaussian_process,
    matern52_kernel,
    robust_cholesky,
)
from causal_active_learning.value_search.grid import GridValueOptimiser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import domain, stub_evaluate

DOUBLE = torch.float64


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


def test_cholesky_retry():
    x = torch.tensor([[0.5], [0.5], [1.0]], dtype=DOUBLE)  # duplicated input
    kernel = matern52_kernel(x, x, 1.0, 1.0)
    with pytest.raises(torch.linalg.LinAlgError):
        robust_cholesky(kernel - 1e-12 * torch.eye(3), jitter=1e-20)
    shifted = kernel - 1.5e-6 * torch.eye(3, dtype=DOUBLE)
    _, used = robust_cholesky(shifted, jitter=1e-6)
    assert used > 1e-6
    gp = condition_gaussian_process(
        x, torch.tensor([0.1, 0.1, 0.3], dtype=DOUBLE), torch.zeros(3),
        GaussianProcessHyperparameters(1.0, 1.0, 0.0),
    )  # fmt: skip
    assert torch.isfinite(gp.posterior(x)[0]).all()


def test_bayesian_optimisation_finds_the_maximum():
    peak = {0: lambda u: math.exp(-((u - 1.3) ** 2) / 0.5)}
    hits = 0
    for seed in range(10):
        result = BayesianOptimisationValueOptimiser().optimise(
            stub_evaluate(peak, 0.02, seed), {0: domain(0)},
            torch.Generator().manual_seed(seed),
        )[0]  # fmt: skip
        assert len(result.evaluations) == 3 + 8
        assert all(-3 <= e.standardised_value <= 3 for e in result.evaluations)
        hits += abs(result.best_value - 1.3) < 0.2
    assert hits >= 9


def test_recommendation_follows_the_posterior_mean():
    def evaluate(candidates):
        out = []
        for c in candidates:
            u = c.intervention_value
            if u == 0.0:  # a large noisy value with a large error
                out.append(InformationGainEstimate(c, 2.0, 1.0, torch.zeros(1)))
            else:
                value = 0.6 - 0.1 * (u - 1.0) ** 2
                out.append(
                    InformationGainEstimate(c, value, 0.01, torch.zeros(1))
                )
        return out

    optimiser = BayesianOptimisationValueOptimiser(initial_points=5, iterations=0)
    result = optimiser.optimise(
        evaluate, {0: domain(0)}, torch.Generator().manual_seed(0)
    )[0]
    raw_best = max(result.evaluations, key=lambda e: e.expected_information_gain)
    assert raw_best.standardised_value == 0.0
    assert result.best_standardised_value == pytest.approx(1.0)


def test_lockstep_equals_separate_runs():
    functions = {
        1: lambda u: math.exp(-((u - 1.0) ** 2)),
        2: lambda u: 0.5 * math.exp(-((u + 2.0) ** 2) / 2),
        4: lambda u: 0.2 + 0.05 * u,
    }
    domains = {n: domain(n) for n in functions}
    optimiser = BayesianOptimisationValueOptimiser(iterations=4, restarts=3)
    calls: list = []
    generator = torch.Generator().manual_seed(11)
    state = generator.get_state()
    together = optimiser.optimise(
        stub_evaluate(functions, 0.02, 0, calls), domains, generator
    )
    assert len(calls) == 3 + 4
    assert all(sorted(c) == [1, 2, 4] for c in calls)
    for node in functions:
        alone_generator = torch.Generator()
        alone_generator.set_state(state)
        alone = optimiser.optimise(
            stub_evaluate(functions, 0.02, 0), {node: domains[node]},
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


def test_bi_level_optimisation_matters():
    width = 0.15
    functions = {
        0: lambda u: 0.3 + 0.6 * math.exp(-((u - 1.6) ** 2) / (2 * width**2)),
        1: lambda u: 0.5,
    }
    domains = {n: domain(n) for n in functions}
    grid = GridValueOptimiser((-2, -1, 0, 1, 2))
    picks_a = 0
    for seed in range(5):
        evaluate = stub_evaluate(functions, 0.01, seed)
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
