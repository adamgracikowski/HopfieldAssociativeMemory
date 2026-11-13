import polars as pl
import pytest

from hopfield_analysis import labels
from hopfield_analysis.benchmark import load_benchmark, medians, plot_benchmark
from hopfield_analysis.cli import main

from .conftest import FIXTURES

BENCHMARK = FIXTURES / "benchmark"
"""``benchmark.toml``: ``small-7x7`` (``N = 49``) and random sets of 3
patterns for ``N = 64`` and 128, dense weights up to ``N = 64`` and factored
weights, both rules, 2 repeats: ``(2 + 2 + 1) * 2 * 2 = 20`` measurements."""


@pytest.fixture
def table() -> pl.DataFrame:
    return load_benchmark(BENCHMARK)


def test_load_benchmark_reads_every_measurement(table):
    assert table.height == 20
    assert table["backend"].unique().sort().to_list() == ["dense", "factored"]
    assert table.filter(pl.col("backend") == "dense")["neurons"].max() == 64
    assert table["train_peak_bytes"].null_count() == 0


def test_medians_have_a_row_per_set_backend_and_rule(table):
    data = medians(table)
    assert data.height == 10
    assert data["repeats"].to_list() == [2] * 10
    first = table.filter(
        (pl.col("source") == "small-7x7")
        & (pl.col("backend") == "dense")
        & (pl.col("learning") == "hebb")
    )
    assert data["sweep_seconds"][0] == pytest.approx(first["sweep_seconds"].median())
    assert data["weights_bytes"][0] == 8 * 49**2


def test_plot_benchmark_draws_lines_markers_and_the_dense_size(table):
    figure = plot_benchmark(table)
    axes = figure.get_axes()
    assert len(axes) == 4
    for ax in axes:
        assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")
    assert len(axes[0].get_lines()) == 4 + 4
    assert len(axes[3].get_lines()) == 4 + 4 + 4 + 1
    assert axes[3].get_ylabel() == labels.MEMORY
    [legend] = figure.legends
    names = [text.get_text() for text in legend.get_texts()]
    assert names[0] == "macierz pełna, Hebb"
    assert names[-3:] == [labels.DENSE_SIZE, "small-7x7", labels.TRAIN_PEAK]


def test_plot_benchmark_rejects_an_empty_table(table):
    with pytest.raises(ValueError, match="no measurements"):
        plot_benchmark(table.clear())


def test_cli_writes_a_pdf_and_a_png(tmp_path, capsys):
    out = tmp_path / "benchmark"
    assert main(["benchmark", str(BENCHMARK), "--out", str(out)]) == 0
    assert out.with_suffix(".pdf").read_bytes()[:5] == b"%PDF-"
    assert out.with_suffix(".png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert main(["benchmark", str(tmp_path), "--out", str(out)]) == 1
    assert "benchmark.parquet" in capsys.readouterr().err
