from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib.patches import StepPatch

from hopfield_analysis.cli import main
from hopfield_analysis.datasets import Registry
from hopfield_analysis.stability import (
    aligned_fields,
    load_datasets,
    load_fields,
    load_stability,
    plot_fields,
    plot_overlaps,
    plot_stable,
    plot_unstable,
    unstable_mask,
    zero_field,
)

from .conftest import FIXTURES

RESULTS = FIXTURES / "experiment"
SHAPES = {"small-7x7": (7, 7), "animals-14x9": (9, 14)}


@pytest.fixture(scope="module")
def registry():
    return Registry.discover(Path(__file__).parent)


@pytest.fixture(scope="module")
def fields():
    return load_fields(RESULTS)


def test_load_fields_has_a_row_per_pattern_and_neuron(fields):
    """Two rules on ``small-7x7`` (5 patterns of 49 neurons) and
    ``animals-14x9`` (6 of 126): ``2 (5 * 49 + 6 * 126) = 2002`` rows."""
    assert fields.columns == ["dataset", "learning", "pattern", "neuron", "field"]
    assert fields.height == 2002
    assert set(fields["learning"]) == {"hebb", "oja (subspace)"}


def test_load_datasets_reads_symmetric_overlaps():
    datasets = load_datasets(RESULTS)
    assert list(datasets) == ["small-7x7", "animals-14x9"]
    small = datasets["small-7x7"]
    assert (small["width"], small["height"], small["patterns"]) == (7, 7, 5)
    overlaps = np.array(small["overlaps"])
    np.testing.assert_allclose(overlaps, overlaps.T)
    np.testing.assert_allclose(np.diag(overlaps), 1.0)
    sums = np.abs(overlaps).sum(axis=1) - 1
    np.testing.assert_allclose(small["overlap_sums"], sums)


def test_load_stability_reads_every_training():
    table = load_stability(RESULTS)
    assert table.select("dataset", "learning", "patterns", "stable").rows() == [
        ("small-7x7", "hebb", 5, 3),
        ("small-7x7", "oja (subspace)", 5, 5),
        ("animals-14x9", "hebb", 6, 2),
        ("animals-14x9", "oja (subspace)", 6, 6),
    ]
    assert table["neurons"].to_list() == [49, 49, 126, 126]
    assert table.filter(pl.col("learning") == "oja (subspace)")["unstable_neurons"].sum() == 0
    assert zero_field(RESULTS) == "keep"


def test_unstable_mask_of_the_fields_matches_the_compute_stability(fields, registry):
    table = load_stability(RESULTS)
    for row in table.iter_rows(named=True):
        patterns = registry.load(row["dataset"])
        aligned = aligned_fields(fields, row["dataset"], row["learning"], SHAPES[row["dataset"]])
        assert aligned.shape == patterns.shape
        mask = unstable_mask(patterns, aligned, "keep")
        assert int(mask.sum()) == row["unstable_neurons"]
        assert int((~mask.any(axis=(1, 2))).sum()) == row["stable"]


def test_unstable_mask_treats_a_zero_field_by_the_setting():
    """Neurons ``+1, -1, +1, -1`` with fields ``0, 0, -0.5, -1e-12``: the
    last field is zero (``|h| <= 1e-10``). ``keep`` changes only the third
    neuron; ``plus-one`` also sets the zero-field ``-1`` neurons to ``+1``."""
    patterns = np.array([1, -1, 1, -1])
    values = np.array([0.0, 0.0, -0.5, -1e-12])
    assert unstable_mask(patterns, values, "keep").tolist() == [False, False, True, False]
    assert unstable_mask(patterns, values, "plus-one").tolist() == [False, True, True, True]
    with pytest.raises(ValueError, match="unknown zero_field"):
        unstable_mask(patterns, values, "minus-one")


def test_aligned_fields_rejects_a_missing_training_or_shape(fields):
    with pytest.raises(ValueError, match="no fields of `small-7x7` / `storkey`"):
        aligned_fields(fields, "small-7x7", "storkey", (7, 7))
    with pytest.raises(ValueError, match="do not fit"):
        aligned_fields(fields, "small-7x7", "hebb", (5, 5))


def test_plot_stable_has_a_bar_per_dataset_and_rule():
    figure = plot_stable(load_stability(RESULTS))
    [ax] = figure.axes
    assert len(ax.patches) == 4
    heights = [bar.get_height() for container in ax.containers for bar in container]
    assert heights == pytest.approx([3 / 5, 2 / 6, 1.0, 1.0])
    texts = [text.get_text() for text in ax.texts]
    assert texts == ["3/5", "2/6", "5/5", "6/6"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["small-7x7", "animals-14x9"]
    [legend] = figure.legends
    assert [t.get_text() for t in legend.get_texts()] == ["Hebb", "Oja (podprzestrzeń)"]


def test_plot_stable_skips_missing_combinations():
    table = pl.DataFrame(
        {
            "dataset": ["a", "a", "b"],
            "learning": ["hebb", "oja (subspace)", "hebb"],
            "patterns": [4, 4, 2],
            "stable": [1, 4, 0],
        }
    )
    [ax] = plot_stable(table).axes
    assert len(ax.patches) == 3


def test_plot_fields_counts_every_field(fields):
    figure = plot_fields(fields)
    axes = [ax for ax in figure.axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == ["small-7x7", "animals-14x9"]
    for ax, neurons in zip(axes, [5 * 49, 6 * 126], strict=True):
        steps = [patch for patch in ax.patches if isinstance(patch, StepPatch)]
        assert [int(step.get_data().values.sum()) for step in steps] == [neurons, neurons]
    assert figure._supxlabel.get_text() == "pole $\\xi_i^\\mu h_i$"


def test_plot_fields_can_use_a_logarithmic_count_axis(fields):
    figure = plot_fields(fields, log=True)
    assert {ax.get_yscale() for ax in figure.axes} == {"log"}
    assert {ax.get_yscale() for ax in plot_fields(fields).axes} == {"linear"}


def test_plot_fields_shows_the_chosen_datasets(fields):
    figure = plot_fields(fields, ["animals-14x9"])
    assert [ax.get_title() for ax in figure.axes] == ["animals-14x9"]
    with pytest.raises(ValueError, match="no fields of `cats`"):
        plot_fields(fields, ["cats"])


def test_plot_unstable_marks_the_masked_pixels():
    patterns = np.ones((3, 2, 4), dtype=np.int8)
    unstable = np.zeros((3, 2, 4), dtype=bool)
    unstable[1, 0, 3] = unstable[2, 1, 0] = unstable[2, 1, 1] = True
    figure = plot_unstable(patterns, unstable)
    assert len(figure.axes) == 3
    for mu, ax in enumerate(figure.axes):
        bitmap, marks = ax.images
        np.testing.assert_array_equal(bitmap.get_array(), patterns[mu])
        np.testing.assert_array_equal(~np.ma.getmaskarray(marks.get_array()), unstable[mu])
    assert [ax.get_xlabel() for ax in figure.axes] == ["0", "1", "2"]
    assert figure._supxlabel.get_text().startswith("liczba niestabilnych pikseli")
    with pytest.raises(ValueError, match="differ in shape"):
        plot_unstable(patterns, unstable[:2])


def test_plot_overlaps_shows_every_matrix():
    datasets = load_datasets(RESULTS)
    figure = plot_overlaps(datasets)
    panels = [ax for ax in figure.axes if ax.get_title()]
    assert [ax.get_title() for ax in panels] == ["small-7x7", "animals-14x9"]
    for ax, count in zip(panels, [5, 6], strict=True):
        [image] = ax.images
        data = np.asarray(image.get_array())
        np.testing.assert_array_equal(data, data.T)
        np.testing.assert_allclose(data, datasets[ax.get_title()]["overlaps"])
        assert len(ax.texts) == count * count
    assert panels[0].texts[1].get_text() == "−0,18"
    with pytest.raises(ValueError, match="no statistics of `x`"):
        plot_overlaps(datasets, ["x"])


def test_plot_overlaps_writes_no_values_into_large_matrices():
    overlaps = np.eye(9).tolist()
    figure = plot_overlaps({"big": {"overlaps": overlaps}})
    assert len(figure.axes[0].texts) == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["stable", str(RESULTS), str(RESULTS)],
        ["fields", str(RESULTS), "--datasets", "small-7x7", "--log"],
        ["unstable", str(RESULTS), "--dataset", "animals-14x9", "--learning", "hebb"],
        ["overlaps", str(RESULTS), "--datasets", "small-7x7,animals-14x9"],
    ],
)
def test_cli_writes_a_pdf_and_a_png(arguments, tmp_path, capsys):
    out = tmp_path / "plots" / arguments[0]
    assert main([*arguments, "--out", str(out)]) == 0
    assert out.with_suffix(".pdf").read_bytes()[:5] == b"%PDF-"
    assert out.with_suffix(".png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert capsys.readouterr().out.count("written to") == 2


def test_cli_rejects_fields_of_another_dataset(tmp_path, capsys):
    arguments = ["unstable", str(RESULTS), "--dataset", "letters-14x20", "--learning", "hebb"]
    assert main([*arguments, "--out", str(tmp_path / "u")]) == 1
    assert "no fields of `letters-14x20` / `hebb`" in capsys.readouterr().err
