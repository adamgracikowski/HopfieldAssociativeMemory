"""Final states of experiments with random inputs: loading and plotting.

An experiment run with ``attractors = true`` writes ``attractors.parquet``:
one row per grid cell and distinct final state, with how many runs ended in
it, its class, energy, overlaps, the closest mixture of stored patterns and
the state itself. All numbers are computed by the compute layer; this module
only reads and plots them.
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from hopfield_analysis import labels, style
from hopfield_analysis.bitmaps import draw_bitmap
from hopfield_analysis.experiments import CLASS_COLORS

ATTRACTORS_FILE = "attractors.parquet"

ENERGY_GROUPS = {
    labels.PATTERN_OR_NEGATION: (["pattern", "negation"], CLASS_COLORS["pattern"]),
    labels.final_class("spurious"): (["spurious"], CLASS_COLORS["spurious"]),
    labels.final_class("cycle"): (["cycle"], CLASS_COLORS["cycle"]),
    labels.final_class("unconverged"): (["unconverged"], CLASS_COLORS["unconverged"]),
}
"""Histograms of :func:`plot_energies`: label, final state classes and color."""


def load_attractors(results: Path) -> pl.DataFrame:
    """Reads the table of distinct final states of an experiment.

    Args:
        results: Directory written by ``compute experiment``.

    Returns:
        One row per cell and distinct final state with the columns described
        in ``src/compute/README.md`` (``dataset``, ``learning``, ``mode``,
        ``rank``, ``count``, ``fraction``, ``class``, ``nearest``,
        ``mixture``, ``state`` and others).

    Raises:
        ValueError: If the experiment was run without ``attractors = true``.
    """
    path = results / ATTRACTORS_FILE
    if not path.is_file():
        raise ValueError(
            f"no {ATTRACTORS_FILE} in `{results}`; run the experiment with `attractors = true`"
        )
    return pl.read_parquet(path)


def bitmap_shapes(metadata: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """The bitmap shape of every dataset of an experiment.

    Args:
        metadata: Metadata from :func:`hopfield_analysis.experiments.load_metadata`.

    Returns:
        ``(height, width)`` by the dataset label in the results (``name:p``
        for a prefix).
    """
    return {
        info["name"] if "prefix" not in info else f"{info['name']}:{info['prefix']}": (
            info["height"],
            info["width"],
        )
        for info in metadata["datasets"]
    }


def _unique(frame: pl.DataFrame, *columns: str) -> list:
    if len(columns) == 1:
        return frame[columns[0]].unique(maintain_order=True).to_list()
    return frame.select(*columns).unique(maintain_order=True).rows()


def _check_datasets(frame: pl.DataFrame, datasets: list[str] | None) -> list[str]:
    known = _unique(frame, "dataset")
    if datasets is None:
        return known
    missing = [dataset for dataset in datasets if dataset not in known]
    if missing:
        raise ValueError(f"no results of `{missing[0]}`")
    return datasets


def _check_mode(frame: pl.DataFrame, mode: str) -> pl.DataFrame:
    cells = frame.filter(pl.col("mode") == mode)
    if cells.is_empty():
        raise ValueError(f"no cells with mode `{mode}`")
    return cells


def describe(row: dict[str, Any]) -> str:
    """Short Polish description of a final state for a panel title.

    Args:
        row: A row of :func:`load_attractors` as a dictionary.

    Returns:
        ``wzorzec 2`` or ``negacja 2`` for a stored pattern or its negation,
        ``mieszanina 3`` for a spurious state equal to a mixture of three
        patterns, otherwise the class name, e.g. ``stan pozorny`` or ``cykl``.
    """
    if row["class"] in ("pattern", "negation"):
        return f"{labels.final_class(row['class'])} {row['nearest']}"
    if row["class"] == "spurious" and row["mixture"] is not None:
        return f"{labels.MIXTURE} {row['mixture']}"
    return labels.final_class(row["class"])


def plot_outcomes(summary: pl.DataFrame) -> Figure:
    """Plots the fractions of the final state classes of every dataset.

    One row of panels per update mode and one column per learning rule; in
    every panel one bar per dataset stacked from the classes ``pattern``,
    ``negation``, ``spurious``, ``cycle`` and ``unconverged``, with the class
    legend above the panels. Meant for random inputs, which give one cell per
    dataset, rule and mode.

    Args:
        summary: Cell aggregates from
            :func:`hopfield_analysis.experiments.load_summary`.

    Returns:
        The figure in the report style with Polish labels.

    Raises:
        ValueError: If a dataset, rule and mode have more than one cell (an
            experiment with damage levels).
    """
    keys = summary.select("dataset", "learning", "mode")
    if keys.is_duplicated().any():
        raise ValueError("more than one cell per dataset, rule and mode (damage levels?)")
    datasets = _unique(summary, "dataset")
    rules = _unique(summary, "learning")
    modes = _unique(summary, "mode")
    figure = style.new_figure(height=1.7 * len(modes) + 1.3)
    axes = figure.subplots(len(modes), len(rules), squeeze=False, sharey=True, sharex=True)
    positions = list(range(len(datasets)))
    for row, mode in enumerate(modes):
        for column, learning in enumerate(rules):
            ax = axes[row, column]
            cells = {
                cell["dataset"]: cell
                for cell in summary.filter(
                    (pl.col("learning") == learning) & (pl.col("mode") == mode)
                ).iter_rows(named=True)
            }
            bottom = [0.0] * len(datasets)
            for name, color in CLASS_COLORS.items():
                fractions = [
                    cells[d][f"classes_{name}_fraction"] if d in cells else 0.0 for d in datasets
                ]
                ax.bar(
                    positions,
                    fractions,
                    bottom=bottom,
                    color=color,
                    width=0.8,
                    label=labels.final_class(name),
                )
                bottom = [b + f for b, f in zip(bottom, fractions, strict=True)]
            ax.set_ylim(0, 1)
            ax.grid(False, axis="x")
            ax.set_xticks(positions, datasets, rotation=40, ha="right", fontsize=7)
            if row == 0:
                ax.set_title(labels.rule(learning))
        axes[row, 0].set_ylabel(f"{labels.mode(mode)}\n{labels.FRACTION}")
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=len(names))
    return figure


def share(fraction: float) -> str:
    """Formats the fraction of runs that ended in a state, with three decimals.

    Args:
        fraction: Fraction of runs, positive.

    Returns:
        The text, e.g. ``0,480``, or ``< 0,001`` for a fraction that would
        round to ``0,000``.
    """
    if round(fraction, 3) == 0:
        return f"< {labels.number(0.001, 3)}"
    return labels.number(fraction, 3)


def plot_attractors(
    attractors: pl.DataFrame, shape: tuple[int, int], dataset: str, top: int = 6
) -> Figure:
    """Shows the most frequent final states of a dataset as bitmaps.

    One row per learning rule and update mode with its ``top`` most frequent
    final states, from the left by rank; every panel is titled with the
    fraction of runs that ended in the state (see :func:`share`) and its
    description (see :func:`describe`).

    Args:
        attractors: Table from :func:`load_attractors`.
        shape: ``(height, width)`` of the bitmaps, e.g. from :func:`bitmap_shapes`.
        dataset: Dataset label in the results.
        top: Number of states in a row.

    Returns:
        The figure in the report style with Polish labels.

    Raises:
        ValueError: If the table has no states of the dataset or a state does
            not have ``height * width`` pixels.
    """
    rows = attractors.filter((pl.col("dataset") == dataset) & (pl.col("rank") < top))
    if rows.is_empty():
        raise ValueError(f"no attractors of `{dataset}`")
    height, width = shape
    series = _unique(rows, "learning", "mode")
    panel = style.TEXT_WIDTH / top
    figure = style.new_figure(height=len(series) * (panel * height / width + 0.45) + 0.1)
    axes = figure.subplots(len(series), top, squeeze=False)
    for line, (learning, mode) in enumerate(series):
        states = rows.filter((pl.col("learning") == learning) & (pl.col("mode") == mode))
        for ax in axes[line, states.height :]:
            ax.set_visible(False)
        for ax, row in zip(axes[line], states.sort("rank").iter_rows(named=True), strict=False):
            state = np.asarray(row["state"], dtype=np.int8)
            if state.size != height * width:
                raise ValueError(
                    f"a state of `{dataset}` has {state.size} pixels, not {height * width}"
                )
            draw_bitmap(ax, state.reshape(height, width))
            ax.set_title(f"{share(row['fraction'])}\n{describe(row)}", fontsize=7)
        axes[line, 0].set_ylabel(f"{labels.rule(learning)}\n{labels.mode(mode)}", fontsize=7)
    return figure


def accessibility(attractors: pl.DataFrame) -> pl.DataFrame:
    """The fraction of runs of every cell that ended in each stored pattern.

    Args:
        attractors: Table from :func:`load_attractors`.

    Returns:
        One row per cell and stored pattern ``mu`` (all ``p`` patterns, also
        those no run reached) with the columns ``dataset``, ``learning``,
        ``mode``, ``pattern`` (``mu``), ``pattern_fraction`` and
        ``negation_fraction`` (runs that ended in the pattern or in its
        negation).
    """
    patterns = (
        attractors.group_by("dataset", "learning", "mode", maintain_order=True)
        .agg(pl.col("overlaps").first().list.len().alias("patterns"))
        .with_columns(pl.int_ranges(0, "patterns", dtype=pl.UInt32).alias("pattern"))
        .explode("pattern")
        .drop("patterns")
    )
    reached = (
        attractors.filter(pl.col("class").is_in(["pattern", "negation"]))
        .group_by("dataset", "learning", "mode", "nearest")
        .agg(
            pl.col("fraction").filter(pl.col("class") == "pattern").sum().alias("pattern_fraction"),
            pl.col("fraction")
            .filter(pl.col("class") == "negation")
            .sum()
            .alias("negation_fraction"),
        )
        .rename({"nearest": "pattern"})
    )
    return patterns.join(
        reached, on=["dataset", "learning", "mode", "pattern"], how="left", maintain_order="left"
    ).fill_null(0.0)


def plot_accessibility(
    attractors: pl.DataFrame, datasets: list[str] | None = None, mode: str = "asynchronous"
) -> Figure:
    """Plots how often random inputs end in each stored pattern and its negation.

    One row of panels per dataset and one column per learning rule; in every
    panel two bars per stored pattern ``mu``: the fraction of runs that ended
    in it and in its negation. Hopfield, Feinstein and Palmer (1983) call it
    the accessibility of a memory.

    Args:
        attractors: Table from :func:`load_attractors`.
        datasets: Datasets to show, in this order; all by default.
        mode: Update mode of the drawn cells.

    Returns:
        The figure in the report style with Polish labels.

    Raises:
        ValueError: If a requested dataset has no results or no cell has the
            mode.
    """
    datasets = _check_datasets(attractors, datasets)
    table = accessibility(_check_mode(attractors, mode))
    rules = _unique(table, "learning")
    figure = style.new_figure(height=1.5 * len(datasets) + 0.7)
    axes = figure.subplots(len(datasets), len(rules), squeeze=False, sharey=True)
    for row, dataset in enumerate(datasets):
        for column, learning in enumerate(rules):
            ax = axes[row, column]
            panel = table.filter(
                (pl.col("dataset") == dataset) & (pl.col("learning") == learning)
            ).sort("pattern")
            mu = panel["pattern"].to_numpy()
            for offset, name in ((-0.2, "pattern"), (0.2, "negation")):
                ax.bar(
                    mu + offset,
                    panel[f"{name}_fraction"].to_numpy(),
                    width=0.4,
                    color=CLASS_COLORS[name],
                    label=labels.final_class(name),
                )
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(False, axis="x")
            if row == 0:
                ax.set_title(labels.rule(learning))
        axes[row, 0].set_ylabel(f"{dataset}\n{labels.FRACTION}")
    figure.supxlabel(labels.PATTERN_INDEX, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        names,
        loc="outside upper center",
        ncols=len(names),
        title=labels.mode(mode),
    )
    return figure


def spurious_counts(attractors: pl.DataFrame) -> pl.DataFrame:
    """The number of distinct spurious fixed points of every cell.

    Args:
        attractors: Table from :func:`load_attractors`.

    Returns:
        One row per cell with the columns ``dataset``, ``learning``, ``mode``
        and ``spurious`` (rows of class ``spurious``, 0 if none).
    """
    return attractors.group_by("dataset", "learning", "mode", maintain_order=True).agg(
        (pl.col("class") == "spurious").sum().cast(pl.Int64).alias("spurious")
    )


def plot_distinct(attractors: pl.DataFrame, summary: pl.DataFrame, sizes: pl.DataFrame) -> Figure:
    """Plots the spurious states against the number of stored patterns.

    One row of panels per dataset in the registry; the left panel shows the
    number of distinct spurious fixed points that the random inputs reached
    (a lower bound of their number in the network, cf. Rojas 1996, tab.
    12.3), the right one the fraction of runs that ended in one, with the
    95% Wilson intervals, both against the number of patterns ``p`` of the
    prefixes. One line per learning rule and update mode.

    Args:
        attractors: Table from :func:`load_attractors`.
        summary: Cell aggregates from
            :func:`hopfield_analysis.experiments.load_summary`.
        sizes: Datasets of the experiment from
            :func:`hopfield_analysis.experiments.dataset_sizes`.

    Returns:
        The figure in the report style with Polish labels.
    """
    counts = spurious_counts(attractors)
    cells = summary.join(counts, on=["dataset", "learning", "mode"], how="left").join(
        sizes, on="dataset", how="inner"
    )
    names = _unique(cells, "name")
    series = _unique(cells, "learning", "mode")
    figure = style.new_figure(height=1.9 * len(names) + 0.8)
    axes = figure.subplots(len(names), 2, squeeze=False, sharex="row")
    for row, name in enumerate(names):
        for learning, mode in series:
            line = cells.filter(
                (pl.col("name") == name)
                & (pl.col("learning") == learning)
                & (pl.col("mode") == mode)
            ).sort("patterns")
            label = f"{labels.rule(learning)}, {labels.mode(mode)}"
            p = line["patterns"].to_numpy()
            axes[row, 0].plot(p, line["spurious"].to_numpy(), marker="o", markersize=3, label=label)
            fraction = line["classes_spurious_fraction"].to_numpy()
            errors = [
                fraction - line["classes_spurious_lower"].to_numpy(),
                line["classes_spurious_upper"].to_numpy() - fraction,
            ]
            axes[row, 1].errorbar(
                p, fraction, yerr=errors, marker="o", markersize=3, capsize=2, label=label
            )
        axes[row, 0].set_yscale("symlog", linthresh=1)
        axes[row, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axes[row, 0].set_ylabel(f"{name}\n{labels.SPURIOUS_COUNT}")
        axes[row, 1].set_ylabel(labels.SPURIOUS_FRACTION)
        axes[row, 1].set_ylim(-0.05, 1.05)
        for ax in axes[row]:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.supxlabel(labels.PATTERN_COUNT, fontsize=9)
    handles, texts = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, texts, loc="outside upper center", ncols=2)
    return figure


def plot_energies(
    trials: pl.DataFrame, datasets: list[str] | None = None, mode: str = "asynchronous"
) -> Figure:
    """Plots histograms of the final energy by final state class.

    One row of panels per dataset and one column per learning rule; in every
    panel one step histogram per group of :data:`ENERGY_GROUPS` that occurs,
    over the same bins, so the energies of stored patterns and of spurious
    states can be compared. At most five ticks on the energy axis, so long
    labels of a narrow range do not overlap. A pattern and its negation have the same energy,
    so they form one group.

    Args:
        trials: Trial table from :func:`hopfield_analysis.experiments.load_trials`.
        datasets: Datasets to show, in this order; all by default.
        mode: Update mode of the drawn runs.

    Returns:
        The figure in the report style with Polish labels.

    Raises:
        ValueError: If a requested dataset has no results or no run has the
            mode.
    """
    datasets = _check_datasets(trials, datasets)
    runs = _check_mode(trials, mode)
    rules = _unique(runs, "learning")
    figure = style.new_figure(height=1.5 * len(datasets) + 0.7)
    axes = figure.subplots(len(datasets), len(rules), squeeze=False)
    for row, dataset in enumerate(datasets):
        for column, learning in enumerate(rules):
            ax = axes[row, column]
            panel = runs.filter((pl.col("dataset") == dataset) & (pl.col("learning") == learning))
            energies = panel["energy"].to_numpy()
            if energies.size == 0:
                continue
            bins = np.histogram_bin_edges(energies, bins=40)
            for group, (classes, color) in ENERGY_GROUPS.items():
                values = panel.filter(pl.col("class").is_in(classes))["energy"].to_numpy()
                if values.size:
                    counts, _ = np.histogram(values, bins=bins)
                    ax.stairs(counts, bins, color=color, label=group)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if row == 0:
                ax.set_title(labels.rule(learning))
        axes[row, 0].set_ylabel(f"{dataset}\n{labels.RUN_COUNT}")
    figure.supxlabel(labels.ENERGY, fontsize=9)
    handles, names = [], []
    for ax in axes.flat:
        for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
            if name not in names:
                handles.append(handle)
                names.append(name)
    figure.legend(
        handles, names, loc="outside upper center", ncols=len(names), title=labels.mode(mode)
    )
    return figure
