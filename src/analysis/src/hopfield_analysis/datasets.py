"""Pattern sets of the dataset registry ``data/datasets/datasets.toml``.

The registry and the pattern files are inputs of the compute layer; this
module reads them in the same format to draw the patterns: a ``.csv`` file has
one pattern per row with the values ``1`` and ``-1`` row by row, a ``.npy``
file holds ``int8`` patterns of shape ``(p, height, width)``.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from hopfield_analysis import labels, style
from hopfield_analysis.bitmaps import draw_bitmap

DEFAULT_REGISTRY = Path("data/datasets/datasets.toml")


@dataclass(frozen=True)
class DatasetEntry:
    """One registered dataset.

    Attributes:
        name: Name in the registry.
        file: Pattern file, resolved against the registry directory.
        width: Bitmap width.
        height: Bitmap height.
        separator: Field separator of a CSV file.
        description: What the patterns show.
    """

    name: str
    file: Path
    width: int
    height: int
    separator: str
    description: str


class Registry:
    """The datasets of a registry file."""

    def __init__(self, path: Path, entries: dict[str, DatasetEntry]) -> None:
        """Creates a registry of ``entries`` read from ``path``."""
        self.path = path
        self.entries = entries

    @classmethod
    def from_file(cls, path: Path) -> "Registry":
        """Reads a registry file.

        Args:
            path: The ``datasets.toml`` file.

        Returns:
            The registry, with pattern files resolved against its directory.
        """
        with path.open("rb") as file:
            datasets = tomllib.load(file)["datasets"]
        entries = {
            name: DatasetEntry(
                name=name,
                file=path.parent / entry["file"],
                width=entry["width"],
                height=entry["height"],
                separator=entry.get("separator", ","),
                description=entry["description"],
            )
            for name, entry in datasets.items()
        }
        return cls(path, entries)

    @classmethod
    def discover(cls, start: Path) -> "Registry":
        """Finds ``data/datasets/datasets.toml`` in ``start`` or a parent.

        Args:
            start: Directory where the search starts.

        Returns:
            The nearest registry.

        Raises:
            FileNotFoundError: If no directory up to the root has a registry.
        """
        start = start.resolve()
        for directory in (start, *start.parents):
            candidate = directory / DEFAULT_REGISTRY
            if candidate.is_file():
                return cls.from_file(candidate)
        raise FileNotFoundError(
            None, f"no {DEFAULT_REGISTRY.as_posix()} in {start} or its parents", str(start)
        )

    def entry(self, name: str) -> DatasetEntry:
        """The entry of dataset ``name``.

        Raises:
            ValueError: If the registry has no such dataset.
        """
        if name not in self.entries:
            known = ", ".join(sorted(self.entries))
            raise ValueError(f"unknown dataset `{name}` in `{self.path}`; known datasets: {known}")
        return self.entries[name]

    def load(self, name: str) -> np.ndarray:
        """Reads the patterns of dataset ``name``.

        Args:
            name: Dataset name.

        Returns:
            ``int8`` patterns of shape ``(p, height, width)``.

        Raises:
            ValueError: If the file has another size or values other than
                ``1`` and ``-1``.
        """
        entry = self.entry(name)
        shape = (entry.height, entry.width)
        if entry.file.suffix == ".npy":
            patterns = np.load(entry.file)
        else:
            rows = np.loadtxt(entry.file, delimiter=entry.separator, dtype=np.int8, ndmin=2)
            if rows.shape[1] != entry.width * entry.height:
                raise ValueError(
                    f"`{entry.file}` has rows of {rows.shape[1]} values, "
                    f"expected {entry.width} x {entry.height}"
                )
            patterns = rows.reshape(-1, *shape)
        if patterns.shape[1:] != shape:
            raise ValueError(f"`{entry.file}` holds {patterns.shape[1:]} bitmaps, expected {shape}")
        if not np.isin(patterns, (-1, 1)).all():
            raise ValueError(f"`{entry.file}` has values other than 1 and -1")
        return patterns.astype(np.int8)


def plot_patterns(
    patterns: np.ndarray,
    columns: int = 6,
    width: float = style.TEXT_WIDTH,
    numbers: bool = False,
) -> Figure:
    """Shows every pattern of a set as a bitmap in a grid.

    A figure narrower than the text width, placed in the report at its own
    size, keeps the 9 pt text of the other figures; the short titles let
    narrow panels sit side by side without overlapping titles.

    Args:
        patterns: Patterns of shape ``(p, height, width)``.
        columns: Largest number of panels in a row.
        width: Figure width in inches, the report text width by default.
        numbers: Whether the panels are titled with the pattern number
            alone instead of ``wzorzec 0``, ``wzorzec 1``, ...

    Returns:
        The figure in the report style, one titled panel per pattern.
    """
    count, height, bitmap_width = patterns.shape
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    panel = width / columns
    figure = style.new_figure(width=width, height=rows * (panel * height / bitmap_width + 0.3))
    axes = figure.subplots(rows, columns, squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    for mu, ax in enumerate(axes.flat[:count]):
        draw_bitmap(ax, patterns[mu])
        ax.set_title(str(mu) if numbers else f"{labels.PATTERN} {mu}")
    return figure
