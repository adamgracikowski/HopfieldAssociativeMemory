import re

import pytest
from matplotlib import font_manager
from matplotlib.ticker import ScalarFormatter
from PIL import Image

from hopfield_analysis import style


def test_new_figure_uses_latin_modern():
    figure = style.new_figure()
    ax = figure.subplots()
    title = ax.set_title("Odsetek odzyskanych wzorców: ąęłóśżźćń")
    path = font_manager.findfont(title.get_fontproperties(), fallback_to_default=False)
    assert "lmroman10-regular" in path
    assert figure.get_size_inches()[0] == pytest.approx(15 / 2.54)


def test_save_writes_a_pdf_with_embedded_latin_modern_and_a_300_dpi_png(tmp_path):
    figure = style.new_figure(width=3, height=2)
    ax = figure.subplots()
    ax.plot([0, 0.5, 1], [1, -0.5, 0])
    ax.set_xlabel("łąka")
    pdf, png = style.save(figure, tmp_path / "out" / "figure.png")
    assert (pdf.name, png.name) == ("figure.pdf", "figure.png")
    fonts = set(re.findall(rb"/BaseFont /\w+\+([\w-]+)", pdf.read_bytes()))
    assert b"LMRoman10-Regular" in fonts
    with Image.open(png) as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=0.01)
        assert image.size == (900, 600)


def test_comma_formatter_uses_a_decimal_comma():
    figure = style.new_figure()
    ax = figure.subplots()
    ax.plot([0, 1], [-0.5, 1])
    style.finish(figure)
    formatter = ax.yaxis.get_major_formatter()
    assert isinstance(formatter, style.CommaFormatter)
    formatter.set_locs([-0.5, 0.0, 0.5, 1.0])
    assert formatter(0.5) == "0,5"
    assert formatter(-0.5) == "−0,5"
    assert formatter(1.0) == "1,0"
    integers = ax.xaxis.get_major_formatter()
    integers.set_locs([1, 2, 3])
    assert integers(2) == "2"


def test_finish_keeps_formatters_that_are_not_scalar():
    figure = style.new_figure()
    ax = figure.subplots()
    ax.bar(["a", "b"], [1, 2])
    category = type(ax.xaxis.get_major_formatter())
    style.finish(figure)
    assert type(ax.xaxis.get_major_formatter()) is category
    assert category is not ScalarFormatter
    assert isinstance(ax.yaxis.get_major_formatter(), style.CommaFormatter)
