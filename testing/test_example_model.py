"""The fixed example model of kshot-example: round-trip through YAML, drawing
one from a data config's prior, and validation of bad files (CPU, no model)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from CITNP.datasets.mixed_regime_generator import MixedRegimeConfig
from CITNP.evaluation import example_model as em

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Five variables: more than the exact oracle enumerates (as in kshot-prior).
CFG_FIVE_VARIABLES = MixedRegimeConfig(
    function_generator="linear_iw",
    alpha0=1.5,
    num_variables=[5],
    graph_type=["ER"],
    graph_degrees={5: [3, 5]},
    normalise=False,
    k_max=2,
    n_ctx_range=(30, 40),
    n_qry=6,
    qry_mode="grouped",
    n_obs_min=10,
)
SEED = 9101
# X1->X2, X2->X3, X3->X5, X1->X4, X4->X5, X1->X3 (0-based source->target).
FIXED_EDGES = [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4), (0, 2)]


def _fixed_graph(D=5, edges=FIXED_EDGES):
    graph = np.zeros((D, D), dtype=int)
    for source, target in edges:
        graph[source, target] = 1
    return graph


def _hand_written_spec():
    """A valid hand-written three-variable chain X1 -> X2 -> X3."""
    return {
        "num_variables": 3,
        "edges": [
            {"source": 0, "target": 1, "weight": 0.9},
            {"source": 1, "target": 2, "weight": -0.4},
        ],
        "noise_variances": [1.0, 0.5, 0.25],
        "do_dist": {"kind": "normal", "scale": 1.5},
        "provenance": {"source": em.HAND_WRITTEN},
    }


# --------------------------------------------------------------------------- #
# Round trip and drawing from the prior
# --------------------------------------------------------------------------- #
def test_draw_write_load_round_trip(tmp_path):
    cfg = CFG_FIVE_VARIABLES
    model = em.draw_example_model(cfg, _fixed_graph(), SEED)
    assert np.array_equal(model.graph, _fixed_graph())
    assert np.array_equal(model.scm.adj, _fixed_graph())
    assert model.provenance["source"] == em.FROM_PRIOR

    path = tmp_path / "example.yaml"
    em.write_example_model(path, model)
    loaded = em.load_example_model(path)
    # identical parameters after a write-then-load round trip
    assert np.array_equal(loaded.scm.B, model.scm.B)
    assert np.array_equal(loaded.scm.s2, model.scm.s2)
    assert np.array_equal(loaded.graph, model.graph)
    assert loaded.do_dist == model.do_dist
    assert loaded.provenance == model.provenance


def test_drawn_weights_follow_the_data_config_prior():
    # the drawn model equals prior_scm with the fixed graph (same mechanisms)
    from CITNP.evaluation import tasks

    model = em.draw_example_model(CFG_FIVE_VARIABLES, _fixed_graph(), SEED)
    scm, graph, do_dist, _ = tasks.prior_scm(
        CFG_FIVE_VARIABLES, 0, SEED, 5, fixed_graph=_fixed_graph()
    )
    assert np.array_equal(model.scm.B, scm.B)
    assert np.array_equal(model.scm.s2, scm.s2)
    assert model.do_dist == do_dist


def test_hand_written_file_loads(tmp_path):
    import yaml

    path = tmp_path / "hand.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(_hand_written_spec(), f, sort_keys=False)
    model = em.load_example_model(path)
    assert model.num_variables == 3
    assert model.scm.B[0, 1] == 0.9 and model.scm.B[1, 2] == -0.4
    assert np.array_equal(model.graph, np.array([
        [0, 1, 0], [0, 0, 1], [0, 0, 0]
    ]))
    # the true effect of X1 on X3 along the chain is 0.9 * -0.4
    assert model.true_effect(0, 2) == pytest.approx(0.9 * -0.4)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s.update(noise_variances=[1.0, 0.5]), "noise_variances must"),
        (lambda s: s.update(noise_variances=[1.0, -0.5, 0.25]), "positive"),
        (
            lambda s: s["edges"].append({"source": 2, "target": 0, "weight": 1.0}),
            "cycle",
        ),
        (
            lambda s: s["edges"].append({"source": 1, "target": 1, "weight": 1.0}),
            "self loop",
        ),
        (
            lambda s: s["edges"].append({"source": 0, "target": 1, "weight": 1.0}),
            "repeated",
        ),
        (
            lambda s: s["edges"].append({"source": 0, "target": 7, "weight": 1.0}),
            "node index outside",
        ),
        (lambda s: s.update(do_dist={"kind": "cauchy"}), "unknown do_dist"),
        (lambda s: s.update(do_dist={"kind": "normal", "scale": 0.0}), "scale"),
        (lambda s: s.pop("edges"), "missing 'edges'"),
    ],
)
def test_invalid_specs_raise_clear_errors(mutate, message):
    spec = _hand_written_spec()
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        em.example_model_from_dict(spec)


def test_to_dict_is_a_round_trip():
    spec = _hand_written_spec()
    model = em.example_model_from_dict(spec)
    again = em.example_model_to_dict(model)
    assert again["num_variables"] == 3
    assert again["noise_variances"] == [1.0, 0.5, 0.25]
    assert again["do_dist"] == {"kind": "normal", "scale": 1.5}
    # edges are the nonzero weights, sorted by source then target
    assert again["edges"] == [
        {"source": 0, "target": 1, "weight": 0.9},
        {"source": 1, "target": 2, "weight": -0.4},
    ]
