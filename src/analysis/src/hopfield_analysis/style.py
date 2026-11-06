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
FONT_TIMESTAMP_LOGGER = "fontTools.ttLib.tables._h_e_a_d"
"""Logger that warns about the old timestamps of the Latin Modern files."""

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
    "pdf.fonttype": 3,
    "savefig.dpi": DPI,
}
"""Style parameters. ``pdf.fonttype`` 3 embeds the glyph outlines: type 42
would put the CFF-based OpenType files into a TrueType font slot, which PDF
tools report as a font type mismatch."""


def apply() -> None:
    """Registers the bundled fonts and sets the style parameters.

    Artists read the parameters when they are created, so this runs before a
    figure is made (see :func:`new_figure`). Calling it again is harmless.
    The warning of fontTools about the timestamps of the font files, issued
    whenever it subsets them for a PDF, is silenced; its other messages stay.
    """
    registered = {entry.fname for entry in font_manager.fontManager.ttflist}
    for path in sorted(FONT_DIR.glob("*.otf")):
        if str(path) not in registered:
            font_manager.fontManager.addfont(str(path))
    logging.getLogger(FONT_TIMESTAMP_LOGGER).setLevel(logging.ERROR)
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

    The formatter keeps its settings (offset, scientific notation, power
    limits): only its class changes, so axes configured by the caller look
    the same apart from the decimal comma.

    Args:
        figure: The figure, after all its axes were drawn.
    """
    for ax in figure.axes:
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if type(formatter) is ScalarFormatter:
                formatter.__class__ = CommaFormatter


def save(figure: Figure, stem: Path) -> list[Path]:
    """Saves ``figure`` as ``<stem>.pdf`` and ``<stem>.png`` (300 dpi).

    Images (bitmaps, heat maps) are resampled differently per format. The
    PDF embeds them unresampled (interpolation ``none``), so every pixel of
    a large bitmap is kept and stays sharp when zoomed in. The PNG uses
    ``auto``: nearest neighbour when a small bitmap is enlarged at least 3
    times, smoothing when a large one is shrunk, which keeps thin lines
    instead of dropping whole rows and columns.

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
    images = [image for ax in figure.axes for image in ax.images]
    interpolations = [image.get_interpolation() for image in images]
    paths = []
    for extension in FORMATS:
        for image in images:
            image.set_interpolation("none" if extension == "pdf" else "auto")
        path = stem.with_name(f"{stem.name}.{extension}")
        figure.savefig(path, dpi=DPI)
        paths.append(path)
    for image, interpolation in zip(images, interpolations, strict=True):
        image.set_interpolation(interpolation)
    return paths
