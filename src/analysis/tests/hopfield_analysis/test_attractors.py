import numpy as np
import polars as pl
import pytest

from hopfield_analysis.attractors import (
    accessibility,
    bitmap_shapes,
    describe,
    load_attractors,
    plot_accessibility,
    plot_attractors,
    plot_distinct,
    plot_energies,
    plot_outcomes,
    share,
    spurious_counts,
)
from hopfield_analysis.cli import main
from hopfield_analysis.experiments import dataset_sizes, load_metadata, load_summary, load_trials

from .conftest import FIXTURES

RANDOM = FIXTURES / "random"
"""``random.toml``: 50 random inputs for ``small-7x7:3`` and ``small-7x7``, both
rules and modes, so 8 cells of 50 runs."""


@pytest.fixture
def attractors() -> pl.DataFrame:
    return load_attractors(RANDOM)


def test_load_attractors_reads_the_table(attractors):
    assert attractors.schema["state"] == pl.List(pl.Int8)
    assert attractors.schema["overlaps"] == pl.List(pl.Float64)
    runs = attractors.group_by("dataset", "learning", "mode").agg(pl.col("count").sum())
    assert runs.height == 8
    assert runs["count"].to_list() == [50] * 8
    assert attractors["state"].list.len().unique().to_list() == [49]
    small = attractors.filter(pl.col("dataset") == "small-7x7")
    assert small["overlaps"].list.len().unique().to_list() == [5]


def test_load_attractors_needs_the_option(tmp_path):
    with pytest.raises(ValueError, match="attractors = true"):
        load_attractors(tmp_path)


def test_class_counts_agree_with_the_summary(attractors):
    summary = load_summary(RANDOM)
    counts = (
        attractors.filter(pl.col("class") == "spurious")
        .group_by("dataset", "learning", "mode")
        .agg(pl.col("count").sum().alias("spurious_runs"))
    )
    joined = summary.join(counts, on=["dataset", "learning", "mode"], how="left").fill_null(0)
    assert joined["spurious_runs"].to_list() == joined["classes_spurious_count"].to_list()
    states = spurious_counts(attractors)
    assert states.height == 8
    assert (states["spurious"] <= joined["spurious_runs"]).all()


def test_bitmap_shapes_use_the_result_labels():
    shapes = bitmap_shapes(load_metadata(RANDOM))
    assert shapes == {"small-7x7:3": (7, 7), "small-7x7": (7, 7)}


def test_describe_names_patterns_mixtures_and_classes():
    row = {"class": "pattern", "nearest": 2, "mixture": None}
    assert describe(row) == "wzorzec 2"
    assert describe({**row, "class": "negation"}) == "negacja 2"
    assert describe({**row, "class": "spurious", "mixture": 3}) == "mieszanina 3"
    assert describe({**row, "class": "spurious"}) == "stan pozorny"
    assert describe({**row, "class": "cycle"}) == "cykl"


def test_share_marks_fractions_below_the_last_decimal():
    assert share(0.4804) == "0,480"
    assert share(0.0005) == "0,001"
    assert share(0.0001) == "< 0,001"


def test_accessibility_has_a_row_per_pattern_of_every_cell(attractors):
    table = accessibility(attractors)
    assert table.height == 4 * 3 + 4 * 5
    reached = attractors.filter(pl.col("class").is_in(["pattern", "negation"]))
    total = table["pattern_fraction"].sum() + table["negation_fraction"].sum()
    assert total == pytest.approx(reached["fraction"].sum())


def test_plot_outcomes_has_a_bar_per_dataset_and_class():
    figure = plot_outcomes(load_summary(RANDOM))
    axes = figure.get_axes()
    assert len(axes) == 4
    assert [ax.get_title() for ax in axes[:2]] == ["Hebb", "Oja (podprzestrzeń)"]
    for ax in axes:
        assert len(ax.patches) == 2 * 5
    assert axes[0].get_ylabel() == "synchroniczny\nodsetek przebiegów"
    [legend] = figure.legends
    assert len(legend.get_texts()) == 5


def test_plot_outcomes_rejects_damage_levels():
    summary = pl.DataFrame(
        {"dataset": ["a", "a"], "learning": ["hebb"] * 2, "mode": ["synchronous"] * 2}
    )
    with pytest.raises(ValueError, match="damage levels"):
        plot_outcomes(summary)


def test_plot_attractors_draws_the_most_frequent_states(attractors):
    figure = plot_attractors(attractors, (7, 7), "small-7x7", top=3)
    axes = figure.get_axes()
    assert len(axes) == 4 * 3
    first = attractors.filter(
        (pl.col("dataset") == "small-7x7")
        & (pl.col("learning") == "hebb")
        & (pl.col("mode") == "synchronous")
        & (pl.col("rank") == 0)
    ).row(0, named=True)
    [image] = axes[0].get_images()
    assert image.get_array().shape == (7, 7)
    assert np.array_equal(image.get_array(), np.reshape(first["state"], (7, 7)))
    assert axes[0].get_title().endswith(describe(first))
    assert axes[0].get_ylabel() == "Hebb\nsynchroniczny"
    with pytest.raises(ValueError, match="no attractors of `x`"):
        plot_attractors(attractors, (7, 7), "x")
    with pytest.raises(ValueError, match="49 pixels"):
        plot_attractors(attractors, (5, 5), "small-7x7")


def test_plot_accessibility_has_two_bars_per_pattern(attractors):
    figure = plot_accessibility(attractors, ["small-7x7"], mode="synchronous")
    axes = figure.get_axes()
    assert len(axes) == 2
    for ax in axes:
        assert len(ax.patches) == 2 * 5
    with pytest.raises(ValueError, match="no results of `x`"):
        plot_accessibility(attractors, ["x"])
    with pytest.raises(ValueError, match="no cells with mode `x`"):
        plot_accessibility(attractors, mode="x")


def test_plot_distinct_has_a_point_per_prefix(attractors):
    metadata = load_metadata(RANDOM)
    figure = plot_distinct(attractors, load_summary(RANDOM), dataset_sizes(metadata))
    left, right = figure.get_axes()
    assert len(left.get_lines()) == 4
    for line in left.get_lines():
        assert list(line.get_xdata()) == [3, 5]
    assert left.get_ylabel() == "small-7x7\nliczba różnych\nstanów pozornych"
    assert right.get_ylim() == (-0.05, 1.05)


def test_plot_energies_draws_a_histogram_per_class_group():
    figure = plot_energies(load_trials(RANDOM), ["small-7x7"])
    axes = figure.get_axes()
    assert len(axes) == 2
    for ax in axes:
        assert len(ax.get_xticks()) <= 5 + 2
    [legend] = figure.legends
    names = [text.get_text() for text in legend.get_texts()]
    assert names[0] == "wzorzec lub negacja"
    assert "negacja" not in names
    assert legend.get_title().get_text() == "asynchroniczny"


@pytest.mark.parametrize(
    "args",
    [
        ["outcomes"],
        ["attractors", "--dataset", "small-7x7:3", "--top", "2"],
        ["accessibility", "--datasets", "small-7x7", "--mode", "synchronous"],
        ["distinct"],
        ["energies", "--datasets", "small-7x7:3"],
    ],
)
def test_cli_writes_a_pdf_and_a_png(args, tmp_path):
    out = tmp_path / args[0]
    assert main([args[0], str(RANDOM), "--out", str(out), *args[1:]]) == 0
    assert out.with_suffix(".pdf").read_bytes()[:5] == b"%PDF-"
    assert out.with_suffix(".png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_reports_missing_attractors_and_datasets(tmp_path, capsys):
    out = str(tmp_path / "f")
    occlusion = str(FIXTURES / "occlusion")
    assert main(["distinct", occlusion, "--out", out]) == 1
    assert "attractors = true" in capsys.readouterr().err
    assert main(["attractors", str(RANDOM), "--dataset", "x", "--out", out]) == 1
    assert "no dataset `x` in the experiment" in capsys.readouterr().err
