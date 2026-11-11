"""Results of ``compute experiment``: loading and plotting.

An experiment directory holds ``trials.parquet`` (one row per run),
``summary.json`` (aggregates of every grid cell) and ``metadata.json``. All
numbers are computed by the compute layer; this module only reads and plots
them.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from hopfield_analysis import labels, style

CLASS_COLORS = {
    "pattern": "C2",
    "negation": "C0",
    "spurious": "C1",
    "cycle": "C4",
    "unconverged": "0.6",
}
"""Colors of the final state classes, in the order of the stacked bars."""

METRICS = {"recovered": "recovered", "nearest-target": "nearest_target"}
"""Recovery metrics of the plots and their fields in ``summary.json``: runs that
ended in their target, and runs whose final state is nearest their target."""

DAMAGE_SOURCES = {"pattern": "noise", "occlusion": "occlusion"}
"""Kinds of damage of the input sources of an experiment with stored patterns."""

PANELS_PER_ROW = 3
"""Largest number of dataset panels in a row of :func:`plot_recovery`."""
TICK_LABELS = 6
"""Largest number of labelled damage levels on an axis of :func:`plot_classes`."""

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


def damage(metadata: dict[str, Any]) -> str:
    """The kind of damage of the inputs of an experiment.

    Args:
        metadata: Metadata from :func:`load_metadata`.

    Returns:
        ``noise`` for flipped pixels (``source = "pattern"``) or ``occlusion``
        for covered rows.

    Raises:
        ValueError: For random inputs, which have no damage.
    """
    source = metadata["config"]["input"]["source"]
    if source not in DAMAGE_SOURCES:
        raise ValueError(f"the experiment has no damaged inputs (source `{source}`)")
    return DAMAGE_SOURCES[source]


def dataset_sizes(metadata: dict[str, Any]) -> pl.DataFrame:
    """The datasets of an experiment with their names in the registry and sizes.

    Args:
        metadata: Metadata from :func:`load_metadata`.

    Returns:
        One row per dataset in configuration order with the columns
        ``dataset`` (the label in the results, ``name:p`` for a prefix),
        ``name`` (in the registry) and ``patterns`` (``p``).
    """
    rows = [
        {
            "dataset": info["name"] if "prefix" not in info else f"{info['name']}:{info['prefix']}",
            "name": info["name"],
            "patterns": info["patterns"],
        }
        for info in metadata["datasets"]
    ]
    return pl.DataFrame(
        rows, schema={"dataset": pl.String, "name": pl.String, "patterns": pl.Int64}
    )


def _metric_column(summary: pl.DataFrame, metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"unknown metric `{metric}`, expected one of {', '.join(METRICS)}")
    column = METRICS[metric]
    if f"{column}_fraction" not in summary.columns:
        raise ValueError(f"the experiment has no `{column}` fractions (random inputs?)")
    return column


def _metric_label(metric: str) -> str:
    return labels.RECOVERED if metric == "recovered" else labels.NEAREST_TARGET


def _damage_label(kind: str, short: bool = False) -> str:
    if kind == "occlusion":
        return labels.OCCLUSION_SHORT if short else labels.OCCLUSION
    return labels.NOISE_SHORT if short else labels.NOISE


def _errorbar(ax, x, series: pl.DataFrame, column: str, label: str) -> None:
    fraction = series[f"{column}_fraction"].to_numpy()
    errors = [
        fraction - series[f"{column}_lower"].to_numpy(),
        series[f"{column}_upper"].to_numpy() - fraction,
    ]
    ax.errorbar(x, fraction, yerr=errors, marker="o", markersize=3, capsize=2, label=label)


def plot_recovery(
    summary: pl.DataFrame, metric: str = "recovered", damage: str = "noise"
) -> Figure:
    """Plots a recovery fraction against the damage level.

    One panel per dataset, at most three in a row, one line per learning
    rule and update mode, with the 95% Wilson intervals from the summary as
    error bars, the legend above and one damage axis label below the panels, all in
    the report style with Polish labels.

    Args:
        summary: Cell aggregates from :func:`load_summary` of an experiment
            with damaged stored patterns as inputs.
        metric: ``recovered`` (runs that ended in their target) or
            ``nearest-target`` (runs whose final state is nearest their
            target, see :data:`METRICS`).
        damage: ``noise`` (flipped pixels) or ``occlusion`` (covered rows),
            e.g. from :func:`damage`; it names the axis.

    Returns:
        The figure, not attached to any GUI backend.

    Raises:
        ValueError: If the metric is unknown or the summary has no fractions
            of it (random inputs).
    """
    column = _metric_column(summary, metric)
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
            label = f"{labels.rule(learning)}, {labels.mode(mode)}"
            _errorbar(ax, series["noise"].to_numpy(), series, column, label)
        ax.set_title(dataset)
        ax.set_ylim(-0.05, 1.05)
    for row in grid:
        row[0].set_ylabel(_metric_label(metric))
    figure.supxlabel(_damage_label(damage), fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=min(len(names), 2))
    return figure


def plot_load(
    summary: pl.DataFrame,
    sizes: pl.DataFrame,
    levels: Sequence[float] | None = None,
    mode: str = "synchronous",
    metric: str = "recovered",
    damage: str = "noise",
) -> Figure:
    """Plots a recovery fraction against the number of stored patterns.

    One row of panels per dataset in the registry and one column per learning
    rule; in every panel one line per damage level against the number of
    patterns ``p`` of the prefixes of the dataset (and of the whole dataset,
    if the experiment has it), with the 95% Wilson intervals as error bars.
    Only the cells of update mode ``mode`` are drawn. The legend of the levels
    is above the panels.

    Args:
        summary: Cell aggregates from :func:`load_summary`.
        sizes: Datasets of the experiment from :func:`dataset_sizes`.
        levels: Damage levels to draw, all of them if ``None``.
        mode: Update mode of the drawn cells.
        metric: ``recovered`` or ``nearest-target``, as in :func:`plot_recovery`.
        damage: ``noise`` or ``occlusion``; it names the legend.

    Returns:
        The figure in the report style with Polish labels.

    Raises:
        ValueError: If the metric is unknown or missing, a requested level is
            not in the summary, or no cell has the mode.
    """
    column = _metric_column(summary, metric)
    cells = summary.filter(pl.col("mode") == mode).join(sizes, on="dataset", how="inner")
    if levels is not None:
        known = set(summary["noise"].drop_nulls().to_list())
        missing = [level for level in levels if level not in known]
        if missing:
            names = ", ".join(labels.decimal(level) for level in missing)
            raise ValueError(f"damage levels not in the experiment: {names}")
        cells = cells.filter(pl.col("noise").is_in(list(levels)))
    if cells.is_empty():
        raise ValueError(f"no cells with mode `{mode}`")
    names = cells["name"].unique(maintain_order=True).to_list()
    rules = cells["learning"].unique(maintain_order=True).to_list()
    drawn = sorted(cells["noise"].unique().to_list())
    figure = style.new_figure(height=1.9 * len(names) + 0.7)
    axes = figure.subplots(len(names), len(rules), squeeze=False, sharey=True, sharex="row")
    for row, name in enumerate(names):
        for col, learning in enumerate(rules):
            ax = axes[row, col]
            panel = cells.filter((pl.col("name") == name) & (pl.col("learning") == learning))
            for level in drawn:
                series = panel.filter(pl.col("noise") == level).sort("patterns")
                _errorbar(ax, series["patterns"].to_numpy(), series, column, labels.decimal(level))
            ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if row == 0:
                ax.set_title(labels.rule(learning))
        axes[row, 0].set_ylabel(f"{name}\n{_metric_label(metric)}")
    figure.supxlabel(labels.PATTERN_COUNT, fontsize=9)
    handles, texts = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        texts,
        loc="outside upper center",
        ncols=min(len(texts), 6),
        title=f"{_damage_label(damage, short=True)} ({labels.mode(mode)})",
    )
    return figure


def plot_classes(summary: pl.DataFrame, damage: str = "noise") -> Figure:
    """Plots the fractions of the final state classes as stacked bars.

    One row of panels per dataset and one column per learning rule and update
    mode; in every panel one bar per damage level (or a single bar for random
    inputs) stacked from the classes ``pattern``, ``negation``, ``spurious``,
    ``cycle`` and ``unconverged``, with the class legend above the panels.
    With more than six damage levels only every k-th bar is labelled, so the
    labels do not overlap.

    Args:
        summary: Cell aggregates from :func:`load_summary`.
        damage: ``noise`` or ``occlusion``; it names the axis.

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
    figure.supxlabel(_damage_label(damage, short=True), fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=len(names))
    return figure
