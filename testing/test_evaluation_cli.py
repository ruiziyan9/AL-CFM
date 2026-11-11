"""Tests for the evaluation CLI's plotting commands and the deprecated module shims."""

import importlib
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import DATA_CFG, GOLDEN, REPO, TINY_CFG

KSHOT_FIGURES = [
    "main",
    "by_query",
    "effect",
    "flag",
    "varratio",
    "qtype",
    "by_t",
]
SHIMS = [
    "eval_d3_kshot",
    "eval_d3_oracle",
    "oracle_bank",
    "oracle_lingauss",
    "plot_kshot_extra",
    "plot_monitoring",
]


@pytest.mark.skipif(
    not (GOLDEN / "kshot.csv").exists(), reason="golden files not recorded"
)
def test_plot_kshot_from_csv(tmp_path):
    from CITNP.evaluation.__main__ import main

    for name in ("kshot.csv", "kshot_by_t.csv"):
        shutil.copy(GOLDEN / name, tmp_path / name)
    main(["plot-kshot", str(tmp_path / "kshot.csv")])
    for fig in KSHOT_FIGURES:
        assert (tmp_path / f"kshot_{fig}.png").stat().st_size > 0, fig


def test_plot_monitoring(tmp_path):
    from CITNP.evaluation.__main__ import main

    run = tmp_path / "run_a"
    run.mkdir()
    (run / "oracle_gap.csv").write_text(
        "step,oracle_gap/all,calib/cover90,flag/logp_gain\n0,0.5,0.8,0.1\n10,0.3,0.9,0.2\n"
    )
    (run / "train_stats.csv").write_text("step,train/loss_mean\n10,1.0\n20,0.8\n")
    out = tmp_path / "monitoring.png"
    main(["plot-monitoring", str(run), "--out", str(out)])
    assert out.stat().st_size > 0


def test_plots_does_not_import_torch():
    import subprocess

    code = "import sys, CITNP.evaluation.plots; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", SHIMS)
def test_shim_warns_and_has_main(name):
    mod = importlib.import_module(f"CITNP.evaluation.{name}")
    with pytest.warns(DeprecationWarning):
        mod = importlib.reload(mod)
    if name != "oracle_lingauss":
        assert callable(mod.main)


def test_oracle_lingauss_reexports():
    from CITNP.evaluation import oracle, tasks

    with pytest.warns(DeprecationWarning):
        old = importlib.reload(
            importlib.import_module("CITNP.evaluation.oracle_lingauss")
        )
    assert old.LinGaussOracle is oracle.LinGaussOracle
    assert old.make_context is tasks.make_context


# --------------------------------------------------------------------------- #
# kshot-prior
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (GOLDEN / "tiny_model.pt").exists(), reason="golden files not recorded"
)
def test_kshot_prior_smoke(tmp_path):
    import json

    import pandas as pd

    from CITNP.evaluation.__main__ import main

    out = tmp_path / "kp.csv"
    main([
        "kshot-prior", "--ckpt", str(GOLDEN / "tiny_model.pt"),
        "--model_config", str(REPO / TINY_CFG),
        "--data_config", str(REPO / DATA_CFG),
        "--n_tasks", "2", "--k_list", "0,1", "--n_mc", "50", "--n_y", "16",
        "--device", "cpu", "--out", str(out),
    ])  # fmt: skip
    df = pd.read_csv(out)
    assert (tmp_path / "kp_by_t.csv").stat().st_size > 0
    assert set(df["design"]) == {"obs", "random", "do(T)", "do(Y)", "do(other)"}
    assert set(df["k"]) == {0, 1} and set(df["task"]) == {0, 1}
    assert df["kl_oracle_model"].notna().all()
    meta = json.loads((tmp_path / "kp_meta.json").read_text())
    assert len(meta["data_config"]["sha256"]) == 64
    assert meta["num_variables"] == [3] and meta["k_list"] == [0, 1]
    assert sum(v["n_tasks"] for v in meta["do_dist_summary"].values()) == 2


# A five-variable data config (built on its parent through `extends`): more
# variables than the exact oracle supports.
DATA_CFG_FIVE_VARIABLES = "CITNP/experiments/configs/oracle_d5_iw_a15.yaml"


@pytest.mark.skipif(
    not (GOLDEN / "tiny_model.pt").exists(), reason="golden files not recorded"
)
def test_kshot_prior_no_oracle_five_variables(tmp_path, capsys):
    import json

    import pandas as pd

    from CITNP.evaluation.__main__ import main
    from CITNP.utils.config_files import config_fingerprint

    data_cfg = str(REPO / DATA_CFG_FIVE_VARIABLES)
    out = tmp_path / "kp.csv"
    args = [
        "kshot-prior", "--ckpt", str(GOLDEN / "tiny_model.pt"),
        "--model_config", str(REPO / TINY_CFG), "--data_config", data_cfg,
        "--n_tasks", "2", "--k_list", "0,1", "--n_y", "16",
        "--device", "cpu", "--out", str(out),
    ]  # fmt: skip
    # with the oracle, five variables are refused, naming the way out
    with pytest.raises(ValueError, match="--no_oracle"):
        main(args)
    assert not out.exists()

    main([*args, "--no_oracle"])
    df = pd.read_csv(out)
    assert set(df["D"]) == {5} and set(df["task"]) == {0, 1}
    assert df["kl_truth_model"].notna().all()
    oracle_columns = [
        "kl_truth_oracle", "kl_oracle_model", "excess_kl_truth",
        "slope_oracle", "var_ratio_model_oracle", "log_prior_true_graph",
        "true_graph_in_prior",
    ]  # fmt: skip
    assert df[oracle_columns].isna().all().all()
    by_t = pd.read_csv(tmp_path / "kp_by_t.csv")
    assert by_t["kl_truth_model"].notna().all()
    assert by_t[["kl_truth_oracle", "mean_oracle"]].isna().all().all()

    meta = json.loads((tmp_path / "kp_meta.json").read_text())
    assert meta["with_oracle"] is False
    assert meta["graph_prior_samples"] is None
    # the fingerprint covers the parent named by `extends`
    assert meta["data_config"]["fingerprint"] == config_fingerprint(data_cfg)
    assert len(meta["data_config"]["sha256"]) == 64
    assert meta["model_config"]["fingerprint"] == config_fingerprint(
        REPO / TINY_CFG
    )

    # the summary skips the oracle's columns, which are entirely missing
    summary = capsys.readouterr().out
    assert "kl_truth_model" in summary and "the oracle was not run" in summary
    assert "kl_truth_oracle" not in summary
    assert "identity check" not in summary


def test_kshot_prior_rejects_foreign_num_variables(tmp_path):
    from CITNP.evaluation.__main__ import main

    with pytest.raises(SystemExit):
        main([
            "kshot-prior", "--ckpt", "x", "--model_config", "y",
            "--data_config", str(REPO / DATA_CFG), "--num_variables", "5",
            "--out", str(tmp_path / "kp.csv"),
        ])  # fmt: skip


def test_plot_kshot_prior_refuses_mixed_data_configs(tmp_path):
    import json

    from CITNP.evaluation.__main__ import main

    paths = []
    for i, h in enumerate(["a" * 64, "b" * 64]):
        csv = tmp_path / f"p_part{i}.csv"
        csv.write_text("D\n3\n")
        (tmp_path / f"p_part{i}_meta.json").write_text(
            json.dumps({"data_config": {"sha256": h}})
        )
        paths.append(str(csv))
    with pytest.raises(SystemExit):
        main(["plot-kshot-prior", *paths])


def _part_meta(**over):
    """A kshot-prior <out>_meta.json as written by the CLI (fields checked
    before merging), with overrides."""
    meta = {
        "data_config": {"path": "d.yaml", "sha256": "a" * 64},
        "model_config": {"path": "m.yaml", "sha256": "b" * 64},
        "ckpt": "model.pt",
        "args": {"base_seed": 9101, "n_mc": 2000, "grid_eps": 0.02},
        "k_list": [0, 1, 3],
        "n_obs": 30,
        "graph_prior_samples": {"3": 20000, "4": 500000},
        "oracle_budget": "total",
    }
    meta.update(over)
    return meta


def test_meta_mismatches_accepts_identical_parts():
    from CITNP.evaluation.__main__ import _meta_mismatches

    assert _meta_mismatches({"p0": _part_meta(), "p1": _part_meta()}) == {}


def test_meta_mismatches_reports_ckpt_and_k_list():
    from CITNP.evaluation.__main__ import _meta_mismatches

    diff = _meta_mismatches({
        "p0": _part_meta(),
        "p1": _part_meta(
            ckpt="other.pt", k_list=[0, 1], oracle_budget="per_dag"
        ),
    })  # fmt: skip
    assert set(diff) == {"ckpt", "k_list", "oracle_budget"}
    assert diff["ckpt"] == {"p0": "model.pt", "p1": "other.pt"}
    assert diff["k_list"] == {"p0": [0, 1, 3], "p1": [0, 1]}


def _part_meta_with_mode(with_oracle: bool, **over):
    """A meta file as written since `--no_oracle` exists: `_part_meta` plus
    `with_oracle` and the config fingerprints."""
    meta = _part_meta(with_oracle=with_oracle)
    meta["data_config"] = {**meta["data_config"], "fingerprint": "c" * 10}
    meta["model_config"] = {**meta["model_config"], "fingerprint": "d" * 10}
    if not with_oracle:
        meta["graph_prior_samples"] = None
    meta.update(over)
    return meta


def test_meta_mismatches_refuses_oracle_and_model_only_parts():
    from CITNP.evaluation.__main__ import _meta_mismatches

    diff = _meta_mismatches({
        "p0": _part_meta_with_mode(True),
        "p1": _part_meta_with_mode(False),
    })  # fmt: skip
    assert diff["with_oracle"] == {"p0": True, "p1": False}
    # a part written before `with_oracle` existed ran the oracle
    old = _part_meta()
    assert "with_oracle" not in old
    assert "with_oracle" in _meta_mismatches({
        "p0": old, "p1": _part_meta_with_mode(False),
    })  # fmt: skip


def test_meta_mismatches_accepts_parts_from_before_with_oracle():
    from CITNP.evaluation.__main__ import _meta_mismatches

    old = _part_meta()
    assert "with_oracle" not in old and "fingerprint" not in old["data_config"]
    assert _meta_mismatches({"p0": old, "p1": _part_meta()}) == {}
    # ... and with a part written now, with the oracle
    assert _meta_mismatches({"p0": old, "p1": _part_meta_with_mode(True)}) == {}


def test_meta_mismatches_compares_config_fingerprints():
    from CITNP.evaluation.__main__ import _meta_mismatches

    # same named file (same sha256), but a parent it extends has changed
    other = _part_meta_with_mode(True)
    other["data_config"] = {**other["data_config"], "fingerprint": "e" * 10}
    diff = _meta_mismatches({"p0": _part_meta_with_mode(True), "p1": other})
    assert set(diff) == {"data_config fingerprint"}
    # an old part records no fingerprint: only the others are compared
    parts = {"old": _part_meta(), "p0": _part_meta_with_mode(True)}
    assert _meta_mismatches(parts) == {}
    assert set(_meta_mismatches({**parts, "p1": other})) == {
        "data_config fingerprint"
    }


def test_meta_mismatches_ignores_oracle_settings_of_model_only_parts():
    from CITNP.evaluation.__main__ import _meta_mismatches

    args = {"base_seed": 9101, "n_mc": 50, "grid_eps": 0.02}
    parts = {
        "p0": _part_meta_with_mode(False),
        "p1": _part_meta_with_mode(False, args=args, oracle_budget="per_dag"),
    }
    assert _meta_mismatches(parts) == {}
    # the same difference matters as soon as the oracle was run
    parts = {
        "p0": _part_meta_with_mode(True),
        "p1": _part_meta_with_mode(True, args=args),
    }
    assert set(_meta_mismatches(parts)) == {"n_mc"}


def test_plot_kshot_prior_refuses_oracle_and_model_only_parts(tmp_path, capsys):
    import json

    from CITNP.evaluation.__main__ import main

    paths = []
    for i, with_oracle in enumerate([True, False]):
        csv = tmp_path / f"p_part{i}.csv"
        csv.write_text("D\n3\n")
        (tmp_path / f"p_part{i}_meta.json").write_text(
            json.dumps(_part_meta_with_mode(with_oracle))
        )
        paths.append(str(csv))
    with pytest.raises(SystemExit):
        main(["plot-kshot-prior", *paths])
    assert "with_oracle" in capsys.readouterr().err


def test_plot_kshot_prior_refuses_mismatched_ckpt(tmp_path, capsys):
    import json

    from CITNP.evaluation.__main__ import main

    paths = []
    for i, ckpt in enumerate(["a.pt", "b.pt"]):
        csv = tmp_path / f"p_part{i}.csv"
        csv.write_text("D\n3\n")
        (tmp_path / f"p_part{i}_meta.json").write_text(
            json.dumps(_part_meta(ckpt=ckpt))
        )
        paths.append(str(csv))
    with pytest.raises(SystemExit):
        main(["plot-kshot-prior", *paths])
    err = capsys.readouterr().err
    assert "ckpt" in err and "a.pt" in err and "b.pt" in err


def test_plot_kshot_prior_writes_figures(tmp_path):
    import itertools
    import json

    import numpy as np
    import pandas as pd

    from CITNP.evaluation.__main__ import main

    rng = np.random.default_rng(0)
    rows = []
    for task, (T, Y) in itertools.product(range(3), [(0, 1), (1, 2)]):
        for design, k in [("obs", 0)] + [
            (d, k) for d in ("random", "do(T)", "do(Y)", "do(other)")
            for k in (1, 3)
        ]:  # fmt: skip
            om = float(rng.uniform(0.0, 0.2))
            to = float(rng.uniform(0.1, 0.5))
            rows.append({
                "D": 3, "task": task, "T": T, "Y": Y,
                "relation": "reverse" if task else "unrelated",
                "design": design, "k": k, "kl_truth_oracle": to,
                "kl_truth_model": to + om, "kl_oracle_model": om,
                "excess_kl_truth": om,
            })  # fmt: skip
    csv = tmp_path / "kp_part0.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    (tmp_path / "kp_part0_meta.json").write_text(
        json.dumps({"data_config": {"sha256": "a" * 64}})
    )
    main(["plot-kshot-prior", str(csv)])
    for s in ("main", "by_relation"):
        assert (tmp_path / f"kp_D3_{s}.png").stat().st_size > 0


def test_plot_kshot_prior_model_only(tmp_path, monkeypatch):
    """Figures from a CSV written without the oracle: its columns are there
    but entirely missing, and the figures say that they show the model."""
    import itertools
    import json

    import numpy as np
    import pandas as pd

    from CITNP.evaluation import plots
    from CITNP.evaluation.__main__ import main

    rng = np.random.default_rng(0)
    rows = []
    for task, (T, Y) in itertools.product(range(3), [(0, 1), (1, 2)]):
        for design, k in [("obs", 0)] + [
            (d, k) for d in ("random", "do(T)", "do(Y)", "do(other)")
            for k in (1, 3)
        ]:  # fmt: skip
            rows.append({
                "D": 5, "task": task, "T": T, "Y": Y,
                "relation": "reverse" if task else "unrelated",
                "design": design, "k": k,
                "kl_truth_model": float(rng.uniform(0.1, 0.7)),
                "kl_truth_oracle": np.nan, "kl_oracle_model": np.nan,
                "excess_kl_truth": np.nan,
            })  # fmt: skip
    csv = tmp_path / "kp_part0.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    (tmp_path / "kp_part0_meta.json").write_text(
        json.dumps({"data_config": {"sha256": "a" * 64}, "with_oracle": False})
    )

    figures = {}  # file name -> (titles, y labels, lines per panel)
    save = plots._save

    def save_and_record(fig, path, dpi=150):
        title = fig._suptitle.get_text() if fig._suptitle else ""
        figures[Path(path).name] = (
            " | ".join([title, *(ax.get_title() for ax in fig.axes)]),
            [ax.get_ylabel() for ax in fig.axes],
            [len(ax.get_lines()) for ax in fig.axes],
        )
        save(fig, path, dpi)

    monkeypatch.setattr(plots, "_save", save_and_record)
    main(["plot-kshot-prior", str(csv)])
    for s in ("main", "by_relation"):
        assert (tmp_path / f"kp_D5_{s}.png").stat().st_size > 0

    titles, ylabels, lines = figures["kp_D5_main.png"]
    assert len(ylabels) == 1  # one panel: no oracle, excess or identity panel
    assert "KL(truth || model)" in titles and "oracle not run" in titles
    assert "KL(truth || model)" in ylabels[0]
    assert lines[0] == 4 + 1  # one curve per design and the line at 0

    titles, ylabels, lines = figures["kp_D5_by_relation.png"]
    assert "KL(truth || model) by question class" in titles
    assert "excess" not in titles and "excess" not in ylabels[0]
    assert "KL(truth || model)" in ylabels[0]
    assert lines == [5, 5]  # two question classes


# --------------------------------------------------------------------------- #
# --fixed_graph_edges
# --------------------------------------------------------------------------- #
def test_parse_graph_edges_valid():
    from CITNP.evaluation.tasks import format_graph_edges, parse_graph_edges

    graph = parse_graph_edges("0->1, 1->2,2->4,0->3,3->4,0->2", 5)
    assert graph.shape == (5, 5) and graph.sum() == 6
    assert graph[0, 1] == graph[2, 4] == graph[0, 2] == 1 and graph[1, 0] == 0
    # canonical form: sorted by source, then target
    assert format_graph_edges(graph) == "0->1,0->2,0->3,1->2,2->4,3->4"


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "empty"),
        ("0->1,,1->2", "bad edge"),
        ("0-1", "bad edge"),
        ("0->a", "bad edge"),
        ("-1->2", "bad edge"),
        ("0->5", "index >= D"),
        ("1->1", "self loop"),
        ("0->1,1->2,0->1", "repeated"),
        ("0->1,1->2,2->0", "cycle"),
    ],
)
def test_parse_graph_edges_rejects(text, message):
    from CITNP.evaluation.tasks import parse_graph_edges

    with pytest.raises(ValueError, match=message):
        parse_graph_edges(text, 5)


def test_cli_rejects_bad_fixed_graph_before_evaluating(tmp_path, capsys):
    from CITNP.evaluation.__main__ import main

    out = tmp_path / "kp.csv"
    args = [
        "kshot-prior", "--ckpt", "unused.pt", "--model_config", "unused.yaml",
        "--data_config", str(REPO / DATA_CFG_FIVE_VARIABLES),
        "--no_oracle", "--out", str(out),
    ]  # fmt: skip
    with pytest.raises(SystemExit):
        main([*args, "--fixed_graph_edges", "0->1,1->0"])
    assert "cycle" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([*args, "--fixed_graph_edges", "0->1,0->1"])
    assert "repeated" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.skipif(
    not (GOLDEN / "tiny_model.pt").exists(), reason="golden files not recorded"
)
def test_kshot_prior_fixed_graph_five_variables(tmp_path):
    import json

    import pandas as pd

    from CITNP.evaluation.__main__ import main

    out = tmp_path / "kp.csv"
    main([
        "kshot-prior", "--ckpt", str(GOLDEN / "tiny_model.pt"),
        "--model_config", str(REPO / TINY_CFG),
        "--data_config", str(REPO / DATA_CFG_FIVE_VARIABLES),
        "--n_tasks", "2", "--k_list", "0,1", "--n_y", "16", "--no_oracle",
        "--fixed_graph_edges", "3->4, 0->1,1->2,2->4,0->3,0->2",
        "--device", "cpu", "--out", str(out),
    ])  # fmt: skip
    df = pd.read_csv(out)
    assert df["fixed_graph"].eq("0->1,0->2,0->3,1->2,2->4,3->4").all()
    meta = json.loads((tmp_path / "kp_meta.json").read_text())
    assert meta["fixed_graph"] == "0->1,0->2,0->3,1->2,2->4,3->4"


def test_meta_mismatches_refuses_different_fixed_graphs():
    from CITNP.evaluation.__main__ import _meta_mismatches

    one = _part_meta_with_mode(True, fixed_graph="0->1,1->2")
    two = _part_meta_with_mode(True, fixed_graph="0->1,0->2")
    assert _meta_mismatches({"p0": one, "p1": one}) == {}
    diff = _meta_mismatches({"p0": one, "p1": two})
    assert set(diff) == {"fixed_graph"}
    assert diff["fixed_graph"] == {"p0": "0->1,1->2", "p1": "0->1,0->2"}


def test_meta_mismatches_refuses_fixed_graph_with_prior_part():
    from CITNP.evaluation.__main__ import _meta_mismatches

    fixed = _part_meta_with_mode(True, fixed_graph="0->1,1->2")
    from_prior = _part_meta_with_mode(True, fixed_graph=None)
    assert set(_meta_mismatches({"p0": fixed, "p1": from_prior})) == {
        "fixed_graph"
    }
    # a meta file from before the option existed is a prior part ...
    old = _part_meta()
    assert "fixed_graph" not in old
    assert set(_meta_mismatches({"p0": fixed, "p1": old})) == {"fixed_graph"}
    # ... and merges with prior parts
    assert _meta_mismatches({"p0": old, "p1": from_prior}) == {}
