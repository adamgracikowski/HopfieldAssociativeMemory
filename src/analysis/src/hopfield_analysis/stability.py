"""Stability of the stored patterns of an experiment: loading and plotting.

``compute experiment`` writes the statistics of every dataset to
``datasets.json`` and, with ``stability_fields = true``, the aligned field
``xi_i^mu h_i`` of every (dataset, rule, pattern, neuron) to
``stability.parquet``; the number of stable patterns of every training is in
``metadata.json``. A neuron of a stored pattern is unstable when its aligned
field is negative; a zero field (``|h| <= 1e-10``) keeps the neuron with
``zero_field = "keep"`` and sets it to ``+1`` with ``"plus-one"``.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from hopfield_analysis import labels, style
from hopfield_analysis.bitmaps import draw_bitmap
from hopfield_analysis.experiments import load_metadata

FIELDS_FILE = "stability.parquet"
DATASETS_FILE = "datasets.json"
ZERO_FIELD = 1e-10
"""Largest absolute field treated as zero, as in the compute layer."""
UNSTABLE_COLOR = "#d62728"
"""Color of the unstable pixels of :func:`plot_unstable`."""
PANELS_PER_ROW = 4
"""Largest number of dataset panels in a row of :func:`plot_fields`."""
OVERLAPS_PER_ROW = 3
"""Largest number of heat maps in a row of :func:`plot_overlaps`."""
ANNOTATED_PATTERNS = 8
"""Largest number of patterns whose overlaps are written into the heat map."""


def load_fields(results: Path) -> pl.DataFrame:
    """Reads the aligned fields of an experiment.

    Args:
        results: Directory written by ``compute experiment`` with
            ``stability_fields = true``.

    Returns:
        One row per (dataset, rule, pattern, neuron) with the columns
        ``dataset``, ``learning``, ``pattern``, ``neuron`` and ``field``.
    """
    return pl.read_parquet(results / FIELDS_FILE)


def load_datasets(results: Path) -> dict[str, dict[str, Any]]:
    """Reads the dataset statistics of an experiment.

    Args:
        results: Directory written by ``compute experiment``.

    Returns:
        The statistics by dataset name: ``width``, ``height``, ``patterns``,
        ``neurons``, ``load``, ``fraction_on``, ``overlaps`` (the matrix
        ``m^{mu nu}``), ``overlap_sums`` and ``max_abs_overlap``.
    """
    text = (results / DATASETS_FILE).read_text(encoding="utf-8")
    return {entry["name"]: entry for entry in json.loads(text)["datasets"]}


def load_stability(results: Path) -> pl.DataFrame:
    """Reads the stability of every training of an experiment.

    Args:
        results: Directory written by ``compute experiment``.

    Returns:
        One row per (dataset, rule) in the order of ``metadata.json`` with
        the columns ``dataset``, ``learning``, ``patterns``, ``neurons``,
        ``stable``, ``unstable_neurons`` (sum over the patterns), ``epochs``
        and ``converged``.
    """
    metadata = load_metadata(results)
    patterns = {entry["name"]: entry["patterns"] for entry in metadata["datasets"]}
    rows = [
        {
            "dataset": training["dataset"],
            "learning": training["learning"],
            "patterns": patterns[training["dataset"]],
            "neurons": training["neurons"],
            "stable": training["stability"]["stable"],
            "unstable_neurons": sum(training["stability"]["unstable_neurons"]),
            "epochs": training["epochs"],
            "converged": training["converged"],
        }
        for training in metadata["trainings"]
    ]
    return pl.DataFrame(rows)


def zero_field(results: Path) -> str:
    """The ``zero_field`` setting of an experiment, ``keep`` or ``plus-one``."""
    return load_metadata(results)["config"]["dynamics"]["zero_field"]


def aligned_fields(
    fields: pl.DataFrame, dataset: str, learning: str, shape: tuple[int, int]
) -> np.ndarray:
    """The aligned fields of one training as bitmaps.

    Args:
        fields: Table from :func:`load_fields`.
        dataset: Dataset name.
        learning: Learning rule name, e.g. ``hebb`` or ``oja (subspace)``.
        shape: Bitmap shape ``(height, width)`` of the dataset.

    Returns:
        Fields of shape ``(p, height, width)``, neurons row by row.

    Raises:
        ValueError: If the table has no fields of the training or not
            ``height * width`` per pattern.
    """
    rows = fields.filter((pl.col("dataset") == dataset) & (pl.col("learning") == learning))
    if rows.height == 0:
        known = ", ".join(
            f"{d} / {r}" for d, r in fields.select("dataset", "learning").unique().sort("*").rows()
        )
        raise ValueError(f"no fields of `{dataset}` / `{learning}`; known: {known}")
    neurons = shape[0] * shape[1]
    if rows.height % neurons != 0:
        raise ValueError(f"{rows.height} fields of `{dataset}` do not fit {shape[1]} x {shape[0]}")
    values = rows.sort("pattern", "neuron")["field"].to_numpy()
    return values.reshape(-1, *shape)


def unstable_mask(patterns: np.ndarray, fields: np.ndarray, zero_field: str) -> np.ndarray:
    """Marks the neurons of the stored patterns that one update would change.

    Args:
        patterns: Patterns with values ``1`` and ``-1``, any shape.
        fields: Aligned fields ``xi_i^mu h_i`` of the same shape.
        zero_field: ``keep`` (a zero field keeps the neuron) or ``plus-one``
            (a zero field sets it to ``+1``, which changes a ``-1``).

    Returns:
        ``True`` where the field is negative or, with ``plus-one``, zero at
        a ``-1`` neuron.

    Raises:
        ValueError: If ``zero_field`` is neither ``keep`` nor ``plus-one``.
    """
    if zero_field not in {"keep", "plus-one"}:
        raise ValueError(f"unknown zero_field `{zero_field}`")
    zero = np.abs(fields) <= ZERO_FIELD
    unstable = (fields < 0) & ~zero
    if zero_field == "plus-one":
        unstable |= zero & (patterns == -1)
    return unstable


def plot_stable(table: pl.DataFrame) -> Figure:
    """Plots the fraction of stable stored patterns as grouped bars.

    One group per dataset, one bar per learning rule, each labelled with
    ``stable/p``; the rule legend is above the panel.

    Args:
        table: Rows of :func:`load_stability`, possibly of several
            experiments concatenated.

    Returns:
        The figure in the report style with Polish labels.
    """
    datasets = table["dataset"].unique(maintain_order=True).to_list()
    rules = table["learning"].unique(maintain_order=True).to_list()
    figure = style.new_figure(height=2.8)
    ax = figure.subplots()
    width = 0.8 / len(rules)
    for k, rule in enumerate(rules):
        rows = {
            row["dataset"]: row
            for row in table.filter(pl.col("learning") == rule).iter_rows(named=True)
        }
        positions = [x for x, d in enumerate(datasets) if d in rows]
        counts = [rows[d] for d in datasets if d in rows]
        offset = (k - (len(rules) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in positions],
            [row["stable"] / row["patterns"] for row in counts],
            width=width,
            color=f"C{k}",
            label=labels.rule(rule),
        )
        ax.bar_label(
            bars,
            [f"{row['stable']}/{row['patterns']}" for row in counts],
            fontsize=6,
            padding=1,
        )
    ax.set_xticks(range(len(datasets)), datasets, rotation=30, ha="right")
    ax.grid(False, axis="x")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(labels.STABLE_FRACTION)
    handles, names = ax.get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=len(names))
    return figure


def plot_fields(
    fields: pl.DataFrame, datasets: list[str] | None = None, log: bool = False
) -> Figure:
    """Plots histograms of the aligned fields.

    One panel per dataset (at most four in a row), one step histogram per
    learning rule over the same bins, and a dashed line at zero: neurons
    left of it are unstable.

    Args:
        fields: Table from :func:`load_fields`.
        datasets: Datasets to show, in this order; all by default.
        log: Logarithmic count axis, so the narrow peak of one rule does not
            flatten the spread of another and single unstable neurons stay
            visible.

    Returns:
        The figure in the report style with Polish labels.

    Raises:
        ValueError: If a requested dataset has no fields.
    """
    if datasets is None:
        datasets = fields["dataset"].unique(maintain_order=True).to_list()
    missing = [d for d in datasets if d not in set(fields["dataset"].to_list())]
    if missing:
        raise ValueError(f"no fields of `{missing[0]}`")
    rules = fields["learning"].unique(maintain_order=True).to_list()
    columns = min(len(datasets), PANELS_PER_ROW)
    rows = math.ceil(len(datasets) / columns)
    figure = style.new_figure(height=1.8 * rows + 0.7)
    grid = figure.subplots(rows, columns, squeeze=False)
    for ax in grid.flat[len(datasets) :]:
        ax.set_visible(False)
    for ax, dataset in zip(grid.flat, datasets, strict=False):
        values = fields.filter(pl.col("dataset") == dataset)
        bins = np.histogram_bin_edges(values["field"].to_numpy(), bins=40)
        for k, rule in enumerate(rules):
            series = values.filter(pl.col("learning") == rule)["field"].to_numpy()
            if series.size:
                counts, _ = np.histogram(series, bins=bins)
                ax.stairs(counts, bins, color=f"C{k}", label=labels.rule(rule))
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.7)
        ax.set_title(dataset)
    for row in grid:
        row[0].set_ylabel(labels.NEURON_COUNT)
    figure.supxlabel(labels.FIELD, fontsize=9)
    handles, names = grid.flat[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=len(names))
    return figure


def plot_unstable(patterns: np.ndarray, unstable: np.ndarray, columns: int = 6) -> Figure:
    """Shows the stored patterns with their unstable pixels in red.

    Args:
        patterns: Patterns of shape ``(p, height, width)``.
        unstable: Mask of the same shape from :func:`unstable_mask`.
        columns: Largest number of panels in a row.

    Returns:
        The figure in the report style, panels titled ``wzorzec 0``, ...
        with the number of unstable pixels below each panel, explained by
        the common label under the grid.

    Raises:
        ValueError: If the shapes differ.
    """
    if patterns.shape != unstable.shape:
        raise ValueError(f"patterns {patterns.shape} and fields {unstable.shape} differ in shape")
    count, height, width = patterns.shape
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    panel = style.TEXT_WIDTH / columns
    figure = style.new_figure(height=rows * (panel * height / width + 0.45))
    axes = figure.subplots(rows, columns, squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    marks = ListedColormap([UNSTABLE_COLOR])
    for mu, ax in enumerate(axes.flat[:count]):
        draw_bitmap(ax, patterns[mu])
        ax.imshow(
            np.ma.masked_where(~unstable[mu], np.ones((height, width))),
            cmap=marks,
            vmin=0,
            vmax=1,
            interpolation="auto",
        )
        ax.set_title(f"{labels.PATTERN} {mu}", fontsize=8)
        ax.set_xlabel(str(int(unstable[mu].sum())), fontsize=8)
    figure.supxlabel(labels.UNSTABLE_PIXELS, fontsize=9)
    return figure


def plot_overlaps(datasets: dict[str, dict[str, Any]], names: list[str] | None = None) -> Figure:
    """Plots the overlap matrices ``m^{mu nu}`` of datasets as heat maps.

    One panel per dataset (at most three in a row) on the common diverging
    scale ``[-1, 1]`` with one color bar; the values are written into the
    cells of sets with at most eight patterns.

    Args:
        datasets: Statistics from :func:`load_datasets`.
        names: Datasets to show, in this order; all by default.

    Returns:
        The figure in the report style with Polish labels.

    Raises:
        ValueError: If a requested dataset is missing.
    """
    if names is None:
        names = list(datasets)
    for name in names:
        if name not in datasets:
            raise ValueError(f"no statistics of `{name}`; known: {', '.join(datasets)}")
    columns = min(len(names), OVERLAPS_PER_ROW)
    rows = math.ceil(len(names) / columns)
    figure = style.new_figure(height=2.3 * rows + 0.2)
    grid = figure.subplots(rows, columns, squeeze=False)
    for ax in grid.flat[len(names) :]:
        ax.set_visible(False)
    image = None
    for ax, name in zip(grid.flat, names, strict=False):
        overlaps = np.array(datasets[name]["overlaps"])
        image = ax.imshow(overlaps, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        count = overlaps.shape[0]
        ticks = range(0, count, max(1, math.ceil(count / 8)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(False)
        ax.set_title(name)
        if count <= ANNOTATED_PATTERNS:
            for (row, column), value in np.ndenumerate(overlaps):
                ax.text(
                    column,
                    row,
                    labels.number(value),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if abs(value) > 0.6 else "black",
                )
    for row in grid:
        row[0].set_ylabel(labels.PATTERN)
    figure.supxlabel(labels.PATTERN, fontsize=9)
    figure.colorbar(image, ax=grid.ravel().tolist(), label=labels.OVERLAP_MATRIX, shrink=0.9)
    return figure
