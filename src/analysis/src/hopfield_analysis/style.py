"""Figure style of the report.

Figures use Latin Modern Roman, the font of the LaTeX report, shipped with the
package in ``fonts/`` (GUST Font License), so they look the same on every
machine without a LaTeX installation. Numbers on axes use the Polish decimal
comma. Every figure is saved as PDF and as PNG at 300 dpi.
"""

import logging
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

FONT_DIR = Path(__file__).parent / "fonts"
FONT_FAMILY = "Latin Modern Roman"
TEXT_WIDTH = 15 / 2.54
"""Text width of the report in inches: A4 with 3 cm side margins."""
DPI = 300
FORMATS = ("pdf", "png")

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": [FONT_FAMILY],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "image.interpolation": "nearest",
    "pdf.fonttype": 42,
    "savefig.dpi": DPI,
}


def apply() -> None:
    """Registers the bundled fonts and sets the style parameters.

    Artists read the parameters when they are created, so this runs before a
    figure is made (see :func:`new_figure`). Calling it again is harmless.
    """
    registered = {entry.fname for entry in font_manager.fontManager.ttflist}
    for path in sorted(FONT_DIR.glob("*.otf")):
        if str(path) not in registered:
            font_manager.fontManager.addfont(str(path))
    # fontTools warns about the old timestamps of the font files when it
    # subsets them for a PDF; the warning is harmless.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    mpl.rcParams.update(RC_PARAMS)


def new_figure(width: float = TEXT_WIDTH, height: float = 2.6) -> Figure:
    """Creates a figure in the report style, with constrained layout.

    Args:
        width: Width in inches, the report text width by default.
        height: Height in inches.

    Returns:
        A figure that is not attached to any GUI backend.
    """
    apply()
    return Figure(figsize=(width, height), layout="constrained")


class CommaFormatter(ScalarFormatter):
    """Tick labels with a decimal comma, e.g. ``0,5`` instead of ``0.5``."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Formats the tick value ``x`` with a decimal comma."""
        return super().__call__(x, pos).replace(".", ",")

    def get_offset(self) -> str:
        """Formats the offset text with a decimal comma."""
        return super().get_offset().replace(".", ",")


def finish(figure: Figure) -> None:
    """Switches every scalar axis of ``figure`` to :class:`CommaFormatter`.

    Args:
        figure: The figure, after all its axes were drawn.
    """
    for ax in figure.axes:
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if type(formatter) is ScalarFormatter:
                axis.set_major_formatter(CommaFormatter())


def save(figure: Figure, stem: Path) -> list[Path]:
    """Saves ``figure`` as ``<stem>.pdf`` and ``<stem>.png`` (300 dpi).

    Args:
        figure: The figure.
        stem: Output path without extension; a ``.pdf`` or ``.png`` extension
            is dropped. Missing parent directories are created.

    Returns:
        The written files.
    """
    if stem.suffix in {f".{extension}" for extension in FORMATS}:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    finish(figure)
    paths = [stem.with_name(f"{stem.name}.{extension}") for extension in FORMATS]
    for path in paths:
        figure.savefig(path, dpi=DPI)
    return paths
