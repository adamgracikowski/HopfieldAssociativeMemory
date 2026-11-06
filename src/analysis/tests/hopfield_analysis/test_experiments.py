import json
from pathlib import Path

import polars as pl
import pytest

from hopfield_analysis.cli import main
from hopfield_analysis.experiments import (
    load_metadata,
    load_summary,
    load_trials,
    plot_classes,
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
    assert figure._supxlabel.get_text() == "poziom szumu (odsetek odwróconych pikseli)"
    assert axes[0].get_ylabel() == "odsetek odzyskanych"
    [legend] = figure.legends
    assert [text.get_text() for text in legend.get_texts()] == [
        "Hebb, synchroniczny",
        "Hebb, asynchroniczny",
        "Oja (podprzestrzeń), synchroniczny",
        "Oja (podprzestrzeń), asynchroniczny",
    ]


def test_plot_recovery_rejects_random_inputs():
    summary = pl.DataFrame({"dataset": ["a"], "noise": [None], "recovered": [None]})
    with pytest.raises(ValueError, match="no recovery fractions"):
        plot_recovery(summary)


def test_cli_writes_a_pdf_and_a_png(results, tmp_path, capsys):
    out = tmp_path / "plots" / "recovery"
    assert main(["recovery", str(results), "--out", str(out)]) == 0
    assert (tmp_path / "plots" / "recovery.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "plots" / "recovery.pdf").read_bytes()[:5] == b"%PDF-"
    assert capsys.readouterr().out.count("written to") == 2


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


def stacked_heights(ax):
    """Sum of the stacked bar heights at every position of ``ax``."""
    totals = {}
    for container in ax.containers:
        for bar in container:
            totals[bar.get_x()] = totals.get(bar.get_x(), 0) + bar.get_height()
    return list(totals.values())


def test_plot_classes_stacks_the_classes_of_every_cell(results):
    figure = plot_classes(load_summary(results))
    axes = [ax for ax in figure.axes]
    assert len(axes) == 2 * 4
    assert axes[0].get_title() == "Hebb\nsynchroniczny"
    assert axes[0].get_ylabel() == "small-7x7\nodsetek przebiegów"
    for ax in axes:
        assert len(ax.containers) == 5
        assert stacked_heights(ax) == pytest.approx([1.0, 1.0])
    assert [tick.get_text() for tick in axes[0].get_xticklabels()] == ["0", "0,2"]
    [legend] = figure.legends
    assert [text.get_text() for text in legend.get_texts()] == [
        "wzorzec",
        "negacja",
        "stan pozorny",
        "cykl",
        "brak zbieżności",
    ]


def test_plot_classes_shows_random_inputs_as_one_bar(results):
    summary = load_summary(results).filter(pl.col("noise") == 0.0)
    summary = summary.with_columns(pl.lit(None, dtype=pl.Float64).alias("noise"))
    figure = plot_classes(summary)
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == ["losowe"]


def test_plot_classes_of_the_example_experiment():
    summary = load_summary(EXAMPLE)
    figure = plot_classes(summary)
    for ax in figure.axes:
        assert stacked_heights(ax) == pytest.approx([1.0] * 4)


def grid_summary(datasets, noise_levels):
    rows = [
        cell(dataset, "hebb", "synchronous", noise, 5)
        for dataset in datasets
        for noise in noise_levels
    ]
    return pl.json_normalize(rows, separator="_", infer_schema_length=None)


def test_plot_recovery_wraps_many_datasets_into_rows():
    summary = grid_summary(["a", "b", "c", "d"], [0.0, 0.1])
    figure = plot_recovery(summary)
    visible = [ax for ax in figure.axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible] == ["a", "b", "c", "d"]
    assert len(figure.axes) == 6
    assert visible[0].get_ylabel() == visible[3].get_ylabel() == "odsetek odzyskanych"


def test_plot_classes_labels_every_kth_of_many_noise_levels():
    """11 levels and at most 6 labels: every ``ceil(11 / 6) = 2``-nd level is
    labelled, with as many digits as the level needs."""
    summary = grid_summary(["a"], [k * 0.025 for k in range(11)])
    figure = plot_classes(summary)
    ticks = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    assert ticks == ["0", "", "0,05", "", "0,1", "", "0,15", "", "0,2", "", "0,25"]
