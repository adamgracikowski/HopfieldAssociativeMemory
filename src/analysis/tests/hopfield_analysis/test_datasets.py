from pathlib import Path

import numpy as np
import pytest

from hopfield_analysis.datasets import Registry, plot_patterns

REPOSITORY = Path(__file__).parents[4]


@pytest.fixture
def registry():
    return Registry.discover(Path(__file__).parent)


def test_finds_the_repository_registry(registry):
    assert registry.path == REPOSITORY / "data" / "datasets" / "datasets.toml"
    assert {"small-7x7", "letters-14x20", "cats-300x200-median"} <= set(registry.entries)
    entry = registry.entry("letters-14x20")
    assert (entry.width, entry.height, entry.separator) == (14, 20, ",")


def test_loads_csv_patterns_row_by_row(registry):
    """The first row of ``small-7x7.csv`` starts with ``1,-1,...,-1,1`` (7
    values) and the first pattern is a diagonal cross, so its first bitmap
    row has ``1`` at both corners."""
    patterns = registry.load("small-7x7")
    assert patterns.shape == (5, 7, 7)
    assert patterns.dtype == np.int8
    assert list(patterns[0, 0]) == [1, -1, -1, -1, -1, -1, 1]
    assert np.array_equal(np.diagonal(patterns[0]), np.ones(7))


def test_loads_npy_patterns_and_checks_the_shape(tmp_path):
    patterns = np.where(np.arange(24).reshape(2, 3, 4) % 3 == 0, 1, -1).astype(np.int8)
    np.save(tmp_path / "set.npy", patterns)
    (tmp_path / "datasets.toml").write_text(
        '[datasets.set]\nfile = "set.npy"\nwidth = 4\nheight = 3\ndescription = "x"\n'
        '[datasets.wrong]\nfile = "set.npy"\nwidth = 3\nheight = 4\ndescription = "x"\n',
        encoding="utf-8",
    )
    registry = Registry.from_file(tmp_path / "datasets.toml")
    assert np.array_equal(registry.load("set"), patterns)
    with pytest.raises(ValueError, match="expected \\(4, 3\\)"):
        registry.load("wrong")


def test_rejects_unknown_datasets_and_invalid_values(tmp_path, registry):
    with pytest.raises(ValueError, match="unknown dataset `nope`.*small-7x7"):
        registry.load("nope")
    (tmp_path / "set.csv").write_text("1,0,1,1\n", encoding="utf-8")
    (tmp_path / "datasets.toml").write_text(
        '[datasets.set]\nfile = "set.csv"\nwidth = 2\nheight = 2\ndescription = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="values other than 1 and -1"):
        Registry.from_file(tmp_path / "datasets.toml").load("set")
    with pytest.raises(FileNotFoundError):
        Registry.discover(tmp_path)


def test_plot_patterns_has_a_titled_panel_per_pattern(registry):
    figure = plot_patterns(registry.load("small-7x7"), columns=3)
    visible = [ax for ax in figure.axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible] == [f"wzorzec {mu}" for mu in range(5)]
    assert len(figure.axes) == 6


def test_plot_patterns_takes_a_width_and_number_titles(registry):
    figure = plot_patterns(registry.load("small-7x7"), columns=5, width=2.5, numbers=True)
    assert figure.get_figwidth() == 2.5
    assert [ax.get_title() for ax in figure.axes] == [str(mu) for mu in range(5)]
