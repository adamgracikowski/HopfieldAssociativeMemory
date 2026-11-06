"""Results of ``compute experiment``: loading and plotting.

An experiment directory holds ``trials.parquet`` (one row per run),
``summary.json`` (aggregates of every grid cell) and ``metadata.json``. All
numbers are computed by the compute layer; this module only reads and plots
them.
"""

import json
import math
from pathlib import Path
from typing import Any

import polars as pl
from matplotlib.figure import Figure

from hopfield_analysis import labels, style

CLASS_COLORS = {
    "pattern": "C2",
    "negation": "C0",
    "spurious": "C1",
    "cycle": "C4",
    "unconverged": "0.6",
}
"""Colors of the final state classes, in the order of the stacked bars."""

PANELS_PER_ROW = 3
"""Largest number of dataset panels in a row of :func:`plot_recovery`."""
TICK_LABELS = 6
"""Largest number of labelled noise levels on an axis of :func:`plot_classes`."""

TRIALS_FILE = "trials.parquet"
SUMMARY_FILE = "summary.json"
METADATA_FILE = "metadata.json"


def load_trials(results: Path) -> pl.DataFrame:
    """Reads the trial table of an experiment.

    Args:
        results: Directory written by ``compute experiment``.

    Returns:
        One row per run with the columns described in ``src/compute/README.md``
        (``dataset``, ``learning``, ``mode``, ``noise``, ``class``,
        ``recovered`` and others).
    """
    return pl.read_parquet(results / TRIALS_FILE)


def load_summary(results: Path) -> pl.DataFrame:
    """Reads the cell aggregates of an experiment as a flat table.

    Nested fields are joined with ``_``, e.g. ``recovered_fraction``,
    ``recovered_lower``, ``classes_pattern_fraction`` or ``steps_mean``.

    Args:
        results: Directory written by ``compute experiment``.

    Returns:
        One row per grid cell, in the order of the grid.
    """
    cells = json.loads((results / SUMMARY_FILE).read_text(encoding="utf-8"))["cells"]
    return pl.json_normalize(cells, separator="_", infer_schema_length=None)


def load_metadata(results: Path) -> dict[str, Any]:
    """Reads the metadata of an experiment.

    Args:
        results: Directory written by ``compute experiment``.

    Returns:
        The configuration, build, datasets and trainings as parsed JSON.
    """
    return json.loads((results / METADATA_FILE).read_text(encoding="utf-8"))


def plot_recovery(summary: pl.DataFrame) -> Figure:
    """Plots the fraction of recovered runs against the noise level.

    One panel per dataset, at most three in a row, one line per learning
    rule and update mode, with the 95% Wilson intervals from the summary as
    error bars, the legend above and one noise axis label below the panels, all in
    the report style with Polish labels.

    Args:
        summary: Cell aggregates from :func:`load_summary` of an experiment
            with damaged stored patterns as inputs.

    Returns:
        The figure, not attached to any GUI backend.

    Raises:
        ValueError: If the summary has no recovery fractions (random inputs).
    """
    if "recovered_fraction" not in summary.columns:
        raise ValueError("the experiment has no recovery fractions (random inputs?)")
    datasets = summary["dataset"].unique(maintain_order=True).to_list()
    columns = min(len(datasets), PANELS_PER_ROW)
    rows = math.ceil(len(datasets) / columns)
    figure = style.new_figure(height=2.1 * rows + 0.7)
    grid = figure.subplots(rows, columns, squeeze=False, sharey=True)
    for ax in grid.flat[len(datasets) :]:
        ax.set_visible(False)
    axes = grid.flat[: len(datasets)]
    for ax, dataset in zip(axes, datasets, strict=True):
        cells = summary.filter(pl.col("dataset") == dataset)
        for (learning, mode), series in cells.group_by(["learning", "mode"], maintain_order=True):
            series = series.sort("noise")
            fraction = series["recovered_fraction"].to_numpy()
            errors = [
                fraction - series["recovered_lower"].to_numpy(),
                series["recovered_upper"].to_numpy() - fraction,
            ]
            ax.errorbar(
                series["noise"].to_numpy(),
                fraction,
                yerr=errors,
                marker="o",
                markersize=3,
                capsize=2,
                label=f"{labels.rule(learning)}, {labels.mode(mode)}",
            )
        ax.set_title(dataset)
        ax.set_ylim(-0.05, 1.05)
    for row in grid:
        row[0].set_ylabel(labels.RECOVERED)
    figure.supxlabel(labels.NOISE, fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=min(len(names), 2))
    return figure


def plot_classes(summary: pl.DataFrame) -> Figure:
    """Plots the fractions of the final state classes as stacked bars.

    One row of panels per dataset and one column per learning rule and update
    mode; in every panel one bar per noise level (or a single bar for random
    inputs) stacked from the classes ``pattern``, ``negation``, ``spurious``,
    ``cycle`` and ``unconverged``, with the class legend above the panels.
    With more than six noise levels only every k-th bar is labelled, so the
    labels do not overlap.

    Args:
        summary: Cell aggregates from :func:`load_summary`.

    Returns:
        The figure in the report style with Polish labels.
    """
    datasets = summary["dataset"].unique(maintain_order=True).to_list()
    series = summary.select("learning", "mode").unique(maintain_order=True).rows()
    figure = style.new_figure(height=1.5 * len(datasets) + 0.6)
    axes = figure.subplots(len(datasets), len(series), squeeze=False, sharey=True)
    for row, dataset in enumerate(datasets):
        for column, (learning, mode) in enumerate(series):
            ax = axes[row, column]
            cells = summary.filter(
                (pl.col("dataset") == dataset)
                & (pl.col("learning") == learning)
                & (pl.col("mode") == mode)
            ).sort("noise")
            positions = list(range(cells.height))
            bottom = [0.0] * cells.height
            for name, color in CLASS_COLORS.items():
                fractions = cells[f"classes_{name}_fraction"].to_list()
                ax.bar(
                    positions,
                    fractions,
                    bottom=bottom,
                    color=color,
                    width=0.8,
                    label=labels.final_class(name),
                )
                bottom = [b + f for b, f in zip(bottom, fractions, strict=True)]
            step = math.ceil(cells.height / TICK_LABELS)
            names = [
                labels.RANDOM_INPUT if n is None else labels.decimal(n)
                for n in cells["noise"].to_list()
            ]
            ax.set_xticks(
                positions, [name if k % step == 0 else "" for k, name in enumerate(names)]
            )
            ax.set_ylim(0, 1)
            ax.grid(False, axis="x")
            if row == 0:
                ax.set_title(f"{labels.rule(learning)}\n{labels.mode(mode)}", fontsize=8)
        axes[row, 0].set_ylabel(f"{dataset}\n{labels.FRACTION}")
    figure.supxlabel(labels.NOISE_SHORT, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=len(names))
    return figure
