"""Tests for tasks.observationally_identified / markov_equivalent_effects and
their relation to the exact oracle's graph uncertainty (D = 3, CPU).

On a large observation-only context the oracle's posterior concentrates on
the Markov equivalence class (MEC) of the true graph. With the
score-equivalent linear_iw prior and a uniform graph prior it is uniform
within the MEC, so its predictive effect should be concentrated at the true
effect exactly for the queries marked identified."""

import sys
from pathlib import Path

import numpy as np

from CITNP.datasets.linear_gaussian_prior import LinearGaussianPrior
from CITNP.evaluation.oracle import LinearSCM, LinGaussOracle, all_dags, dag_key
from CITNP.evaluation.tasks import (
    markov_equivalent_effects,
    observationally_identified,
    prior_scm,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_intervention_probes import CFG  # noqa: E402  (linear_iw, D = 3)

D = 3
BASE_SEED = 4321
NUM_OBSERVATIONS = 1_000_000  # leak to denser graphs ~ n^(-1/2)
MIN_STANDARDISED_WEIGHT = 0.2  # faithfulness margin of the selected SCMs
NUM_MC = 400
FAR = 0.1  # standardised slope distance from the true effect
QUERIES_PER_CLASS = 3
TASK_SEARCH_LIMIT = 60


def test_explicit_graphs():
    chain = np.zeros((3, 3))
    chain[0, 1], chain[1, 2] = 0.9, -0.8
    chain_scm = LinearSCM(chain, np.array([0.3, 0.2, 0.25]))
    # MEC {0->1->2, 2->1->0, 0<-1->2}: the effect of X0 on X2 is not identified
    assert not observationally_identified(chain_scm, 0, 2)
    assert len(markov_equivalent_effects(chain_scm, 0, 2)) == 3
    collider = np.zeros((3, 3))
    collider[0, 2], collider[1, 2] = 0.9, 0.7
    collider_scm = LinearSCM(collider, np.array([0.3, 0.3, 0.2]))
    # a v-structure is alone in its MEC: every query is identified
    assert observationally_identified(collider_scm, 0, 2)
    assert observationally_identified(collider_scm, 2, 0)
    assert len(markov_equivalent_effects(collider_scm, 0, 2)) == 1


def test_true_dag_is_one_of_the_equivalent_effects():
    scm, _, _, _ = prior_scm(CFG, 0, BASE_SEED, D)
    for T, Y in [(0, 1), (2, 0)]:
        effects = markov_equivalent_effects(scm, T, Y)
        truth = np.array(scm.do_dist(T, Y))
        assert np.isclose(effects, truth, rtol=1e-8, atol=1e-10).all(1).any()


def standard_deviations(scm: LinearSCM) -> np.ndarray:
    A = np.linalg.inv(np.eye(D) - scm.B.T)
    return np.sqrt(np.diag(A @ np.diag(scm.s2) @ A.T))


def is_clearly_faithful(scm: LinearSCM) -> bool:
    """Every edge's standardised weight exceeds MIN_STANDARDISED_WEIGHT."""
    sd = standard_deviations(scm)
    standardised = np.abs(scm.B) * sd[:, None] / sd[None, :]
    return bool((standardised[scm.B != 0] > MIN_STANDARDISED_WEIGHT).all())


def test_oracle_is_certain_exactly_for_identified_queries():
    """Large observation-only context: the oracle's mixture puts little mass
    on interventional slopes far (> FAR, standardised) from the true one for
    identified queries, and a clear share for unidentified queries whose
    Markov-equivalent effects differ by more than 3 FAR."""
    dags = all_dags(D)
    uniform = {dag_key(g): -np.log(len(dags)) for g in dags}
    parameter_prior = LinearGaussianPrior.from_config(CFG)
    far_mass = {True: [], False: []}
    for i in range(TASK_SEARCH_LIMIT):
        if min(len(v) for v in far_mass.values()) >= QUERIES_PER_CLASS:
            break
        scm, _, _, _ = prior_scm(CFG, i, BASE_SEED, D)
        if not is_clearly_faithful(scm):
            continue
        rng = np.random.default_rng([BASE_SEED, i])
        x, mask = scm.sample(NUM_OBSERVATIONS, None, None, rng)
        assert not mask.any()  # observation-only context
        oracle = LinGaussOracle(D, log_graph_prior=uniform, prior=parameter_prior)
        oracle.fit(x, mask)
        sd = standard_deviations(scm)
        for T in range(D):
            for Y in range(D):
                if T == Y:
                    continue
                to_standardised = sd[T] / sd[Y]
                identified = observationally_identified(scm, T, Y)
                effect_range = np.ptp(markov_equivalent_effects(scm, T, Y)[:, 0])
                if not identified and effect_range * to_standardised <= 3 * FAR:
                    continue  # effects too close to tell apart
                log_w, slope, _ = oracle.predictive_components(
                    T, Y, n_mc=NUM_MC, min_graph_prob=1e-8, seed=i,
                    budget="total",
                )  # fmt: skip
                w = np.exp(log_w - np.logaddexp.reduce(log_w))
                true_slope = scm.do_dist(T, Y)[0]
                far = np.abs(slope - true_slope) * to_standardised > FAR
                far_mass[identified].append(float(w[far].sum()))
    assert len(far_mass[True]) >= QUERIES_PER_CLASS, far_mass
    assert len(far_mass[False]) >= QUERIES_PER_CLASS, far_mass
    assert max(far_mass[True]) < 0.05, far_mass
    # uniform over an MEC of at most 6 DAGs: a differing member has >= 1/6
    assert min(far_mass[False]) > 0.1, far_mass
