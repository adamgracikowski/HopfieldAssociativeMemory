import json
import math

import numpy as np
import polars as pl
import pytest

from hopfield_analysis.capacity import load_capacity, load_capacity_summary, plot_capacity
from hopfield_analysis.cli import main

from .conftest import FIXTURES

RESULTS = FIXTURES / "capacity"


def test_load_capacity_has_a_row_per_set_and_rule():
    """Prefixes of 1 to 5 patterns of ``small-7x7`` and 10 values of ``p``
    with 5 random sets each, two rules: ``2 (5 + 50) = 110`` rows."""
    table = load_capacity(RESULTS)
    assert table.height == 110
    prefixes = table.filter(pl.col("source") == "small-7x7")
    assert prefixes["seed"].null_count() == prefixes.height
    random = table.filter(pl.col("source") == "random")
    assert random["seed"].null_count() == 0
    assert random.filter(pl.col("learning") == "oja (subspace)")["converged"].all()


def test_load_capacity_summary_has_the_theory_of_random_hebbian_sets():
    summary = load_capacity_summary(RESULTS)
    assert summary.height == 2 * (5 + 10)
    assert "theory" not in summary.columns
    with_theory = summary.filter(pl.col("theory_pattern_stable").is_not_null())
    assert set(with_theory.select("source", "learning").rows()) == {("random", "hebb")}
    assert with_theory.height == 10


def test_theory_matches_the_normal_approximation():
    """``P_bit = Phi(-sqrt(N / p)) = erfc(sqrt(N / p) / sqrt(2)) / 2`` and a
    pattern of ``N`` neurons is stable with probability ``(1 - P_bit)^N``."""
    summary = load_capacity_summary(RESULTS).filter(pl.col("theory_bit_error").is_not_null())
    for row in summary.iter_rows(named=True):
        n, p = row["neurons"], row["patterns"]
        bit = 0.5 * math.erfc(math.sqrt(n / p) / math.sqrt(2))
        assert row["theory_bit_error"] == pytest.approx(bit, rel=1e-12)
        assert row["theory_pattern_stable"] == pytest.approx((1 - bit) ** n, rel=1e-12)


def test_load_capacity_summary_adds_missing_theory_columns(tmp_path):
    rows = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["rows"]
    prefixes = [row for row in rows if row["source"] != "random"]
    (tmp_path / "summary.json").write_text(json.dumps({"rows": prefixes}), encoding="utf-8")
    summary = load_capacity_summary(tmp_path)
    assert summary["theory_pattern_stable"].null_count() == summary.height
    assert summary["theory_bit_error"].dtype == pl.Float64


def test_plot_capacity_has_a_line_per_source_and_rule():
    summary = load_capacity_summary(RESULTS)
    figure = plot_capacity(summary)
    top, bottom = figure.axes
    assert top.get_title() == "$N = 49$"
    names = [line.get_label() for line in top.get_lines()]
    assert names == [
        "small-7x7, Hebb",
        "small-7x7, Oja (podprzestrzeń)",
        "wzorce losowe, Hebb",
        "wzorce losowe, Oja (podprzestrzeń)",
        "teoria (wzorce losowe, Hebb)",
        "$p/N = 0{,}138$",
        "$p/N = 1/(2 \\ln N)$",
    ]
    hebb = summary.filter((pl.col("source") == "random") & (pl.col("learning") == "hebb"))
    line = top.get_lines()[2]
    np.testing.assert_allclose(line.get_xdata(), hebb["load"].to_numpy())
    np.testing.assert_allclose(line.get_ydata(), hebb["stable_fraction_mean"].to_numpy())
    np.testing.assert_allclose(
        bottom.get_lines()[2].get_ydata(), hebb["bit_errors_mean"].to_numpy()
    )
    np.testing.assert_allclose(
        top.get_lines()[4].get_ydata(), hebb["theory_pattern_stable"].to_numpy()
    )
    assert bottom.get_lines()[6].get_xdata()[0] == pytest.approx(1 / (2 * math.log(49)))
    [legend] = figure.legends
    assert [text.get_text() for text in legend.get_texts()] == names


def test_plot_capacity_has_a_column_per_number_of_neurons():
    def rows(source, neurons, theory):
        return [
            {
                "source": source,
                "learning": "hebb",
                "neurons": neurons,
                "patterns": p,
                "load": p / neurons,
                "stable_fraction_mean": 1.0,
                "bit_errors_mean": 0.0,
                "theory_pattern_stable": 1.0 if theory else None,
                "theory_bit_error": 0.0 if theory else None,
            }
            for p in (1, 2)
        ]

    summary = pl.DataFrame(
        rows("random", 625, True) + rows("letters-14x20", 280, False) + rows("random", 280, True)
    )
    figure = plot_capacity(summary)
    assert [ax.get_title() for ax in figure.axes[:2]] == ["$N = 280$", "$N = 625$"]
    left, right = figure.axes[0], figure.axes[1]
    assert [line.get_label() for line in left.get_lines()][:2] == [
        "letters-14x20, Hebb",
        "wzorce losowe, Hebb",
    ]
    assert left.get_lines()[1].get_color() == right.get_lines()[0].get_color()
    [legend] = figure.legends
    assert len(legend.get_texts()) == 5


def test_cli_writes_a_pdf_and_a_png(tmp_path, capsys):
    out = tmp_path / "plots" / "capacity"
    assert main(["capacity", str(RESULTS), "--out", str(out)]) == 0
    assert out.with_suffix(".pdf").read_bytes()[:5] == b"%PDF-"
    assert out.with_suffix(".png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert capsys.readouterr().out.count("written to") == 2


def test_cli_reports_a_missing_run(tmp_path, capsys):
    assert main(["capacity", str(tmp_path), "--out", str(tmp_path / "c")]) == 1
    assert "summary.json" in capsys.readouterr().err
