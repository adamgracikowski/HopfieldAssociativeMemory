import json
from pathlib import Path

import polars as pl
import pytest

from hopfield_analysis.cli import main
from hopfield_analysis.experiments import (
    load_metadata,
    load_summary,
    load_trials,
    plot_recovery,
)

EXAMPLE = Path(__file__).parents[4] / "data" / "experiments" / "00-example" / "results"


def proportion(count, runs):
    """A proportion in the format of summary.json; the interval is not
    checked by these tests, so it is set to the fraction itself."""
    fraction = count / runs
    return {"count": count, "fraction": fraction, "lower": fraction, "upper": fraction}


def cell(dataset, learning, mode, noise, recovered):
    runs = 10
    classes = {
        name: proportion(count, runs)
        for name, count in [
            ("pattern", recovered),
            ("negation", 0),
            ("spurious", runs - recovered),
            ("cycle", 0),
            ("unconverged", 0),
        ]
    }
    return {
        "dataset": dataset,
        "learning": learning,
        "mode": mode,
        "noise": noise,
        "flips": round(noise * 49),
        "runs": runs,
        "classes": classes,
        "recovered": proportion(recovered, runs),
        "steps": {"mean": 1.5, "std": 0.5},
        "energy": {"mean": -20.0, "std": None},
        "final_overlap": {"mean": 0.9, "std": 0.1},
    }


@pytest.fixture
def results(tmp_path):
    cells = [
        cell(dataset, learning, mode, noise, recovered)
        for dataset in ["small-7x7", "animals-14x9"]
        for learning in ["hebb", "oja (subspace)"]
        for mode in ["synchronous", "asynchronous"]
        for noise, recovered in [(0.0, 10), (0.2, 7)]
    ]
    (tmp_path / "summary.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"command": "experiment", "config": {"trials": 2}}), encoding="utf-8"
    )
    pl.DataFrame(
        {
            "dataset": ["small-7x7", "small-7x7"],
            "class": ["pattern", "spurious"],
            "recovered": [True, None],
            "seed": pl.Series([1, 2**63 + 5], dtype=pl.UInt64),
        }
    ).write_parquet(tmp_path / "trials.parquet")
    return tmp_path


def test_load_summary_flattens_nested_fields(results):
    summary = load_summary(results)
    assert summary.height == 16
    assert summary.row(1, named=True)["recovered_fraction"] == pytest.approx(0.7)
    assert summary["classes_spurious_count"].to_list()[:2] == [0, 3]
    assert summary["steps_mean"][0] == 1.5
    assert summary["energy_std"].null_count() == 16


def test_load_trials_and_metadata(results):
    trials = load_trials(results)
    assert trials.schema["seed"] == pl.UInt64
    assert trials["seed"][1] == 2**63 + 5
    assert trials["recovered"].to_list() == [True, None]
    assert load_metadata(results)["config"]["trials"] == 2


def test_plot_recovery_has_a_panel_per_dataset_and_a_line_per_series(results):
    figure = plot_recovery(load_summary(results))
    axes = figure.get_axes()
    assert [ax.get_title() for ax in axes] == ["small-7x7", "animals-14x9"]
    for ax in axes:
        assert len(ax.get_lines()) >= 4
        labels = ax.get_legend_handles_labels()[1]
        assert labels == [
            "hebb, synchronous",
            "hebb, asynchronous",
            "oja (subspace), synchronous",
            "oja (subspace), asynchronous",
        ]


def test_plot_recovery_rejects_random_inputs():
    summary = pl.DataFrame({"dataset": ["a"], "noise": [None], "recovered": [None]})
    with pytest.raises(ValueError, match="no recovery fractions"):
        plot_recovery(summary)


def test_cli_writes_a_png(results, tmp_path, capsys):
    out = tmp_path / "plots" / "recovery.png"
    assert main(["recovery", str(results), "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "written to" in capsys.readouterr().out


def test_reads_the_results_of_the_example_experiment():
    trials = load_trials(EXAMPLE)
    summary = load_summary(EXAMPLE)
    metadata = load_metadata(EXAMPLE)
    config = metadata["config"]
    expected_cells = (
        len(config["datasets"])
        * len(config["learning"])
        * len(config["dynamics"]["modes"])
        * len(config["input"]["noise"])
    )
    assert summary.height == expected_cells
    assert summary["runs"].sum() == trials.height
    assert set(trials["class"].unique()) <= {
        "pattern",
        "negation",
        "spurious",
        "cycle",
        "unconverged",
    }
    figure = plot_recovery(summary)
    assert len(figure.get_axes()) == len(config["datasets"])
