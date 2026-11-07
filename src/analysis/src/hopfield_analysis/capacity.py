"""Results of ``compute capacity``: loading and plotting.

A capacity directory holds ``capacity.parquet`` (one row per pattern set and
learning rule), ``summary.json`` (aggregates of every source, number of
patterns and rule, with the prediction for random Hebbian sets) and
``metadata.json``. All numbers are computed by the compute layer; this
module only reads and plots them.
"""

import json
import math
from pathlib import Path

import polars as pl
from matplotlib.figure import Figure

from hopfield_analysis import labels, style

CAPACITY_FILE = "capacity.parquet"
SUMMARY_FILE = "summary.json"
RANDOM_SOURCE = "random"
"""Source name of random pattern sets."""
HOPFIELD_LOAD = 0.138
"""Load ``p / N`` at which retrieval of random patterns breaks down
(Amit, Gutfreund and Sompolinsky 1985)."""
THEORY_COLUMNS = ("theory_bit_error", "theory_pattern_stable")


def load_capacity(results: Path) -> pl.DataFrame:
    """Reads the table of a capacity run.

    Args:
        results: Directory written by ``compute capacity``.

    Returns:
        One row per pattern set and learning rule with the columns described
        in ``src/compute/README.md`` (``source``, ``learning``, ``neurons``,
        ``patterns``, ``trial``, ``seed``, ``stable``, ``unstable_neurons``,
        ``epochs``, ``converged``).
    """
    return pl.read_parquet(results / CAPACITY_FILE)


def load_capacity_summary(results: Path) -> pl.DataFrame:
    """Reads the aggregates of a capacity run as a flat table.

    Nested fields are joined with ``_``, e.g. ``stable_fraction_mean``,
    ``all_stable_lower`` or ``theory_pattern_stable``; the theory columns
    are null where the run has no prediction and present even if no row has
    one.

    Args:
        results: Directory written by ``compute capacity``.

    Returns:
        One row per (source, ``p``, rule), in the order of the run.
    """
    rows = json.loads((results / SUMMARY_FILE).read_text(encoding="utf-8"))["rows"]
    summary = pl.json_normalize(rows, separator="_", infer_schema_length=None)
    if "theory" in summary.columns:
        summary = summary.drop("theory")
    return summary.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(column)
        for column in THEORY_COLUMNS
        if column not in summary.columns
    )


def source_label(source: str) -> str:
    """Polish name of a pattern source: the dataset name or ``wzorce losowe``."""
    return labels.RANDOM_PATTERNS if source == RANDOM_SOURCE else source


def plot_capacity(summary: pl.DataFrame) -> Figure:
    """Plots the stability of the stored patterns against the load ``p / N``.

    One column of panels per number of neurons ``N``: above the mean
    fraction of stable patterns of a set, below the mean fraction of
    unstable neurons. One line per source and learning rule (the same color
    in every panel), the prediction for random Hebbian sets as a black dashed
    line, and vertical lines at ``p / N = 0.138`` and ``1 / (2 ln N)``.

    Args:
        summary: Aggregates from :func:`load_capacity_summary`.

    Returns:
        The figure in the report style with Polish labels.
    """
    sizes = sorted(summary["neurons"].unique().to_list())
    series = summary.select("source", "learning").unique(maintain_order=True).rows()
    colors = {key: f"C{k}" for k, key in enumerate(series)}
    figure = style.new_figure(height=4.2)
    axes = figure.subplots(2, len(sizes), squeeze=False, sharex="col")
    for column, neurons in enumerate(sizes):
        top, bottom = axes[0, column], axes[1, column]
        cells = summary.filter(pl.col("neurons") == neurons)
        for (source, learning), rows in cells.group_by(["source", "learning"], maintain_order=True):
            rows = rows.sort("patterns")
            name = f"{source_label(source)}, {labels.rule(learning)}"
            style_args = {
                "color": colors[(source, learning)],
                "marker": "o" if source != RANDOM_SOURCE else "s",
                "markersize": 2.5,
                "linewidth": 1,
                "label": name,
            }
            load = rows["load"].to_numpy()
            top.plot(load, rows["stable_fraction_mean"].to_numpy(), **style_args)
            bottom.plot(load, rows["bit_errors_mean"].to_numpy(), **style_args)
        theory = cells.filter(pl.col("theory_pattern_stable").is_not_null()).sort("patterns")
        if theory.height:
            load = theory["load"].to_numpy()
            line = {"color": "black", "linestyle": "--", "linewidth": 0.9, "label": labels.THEORY}
            top.plot(load, theory["theory_pattern_stable"].to_numpy(), **line)
            bottom.plot(load, theory["theory_bit_error"].to_numpy(), **line)
        for ax in (top, bottom):
            ax.axvline(
                HOPFIELD_LOAD, color="0.4", linestyle=":", linewidth=0.9, label=labels.HOPFIELD_LOAD
            )
            ax.axvline(
                1 / (2 * math.log(neurons)),
                color="0.4",
                linestyle="-.",
                linewidth=0.9,
                label=labels.STABLE_LOAD,
            )
        top.set_title(f"$N = {neurons}$")
        top.set_ylim(-0.05, 1.05)
        bottom.set_ylim(bottom=0)
    axes[0, 0].set_ylabel(labels.STABLE_FRACTION_WRAPPED)
    axes[1, 0].set_ylabel(labels.BIT_ERRORS_WRAPPED)
    figure.supxlabel(labels.LOAD, fontsize=9)
    entries = {}
    for ax in axes[0]:
        for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(name, handle)
    figure.legend(
        list(entries.values()),
        list(entries),
        loc="outside upper center",
        ncols=min(len(entries), 3),
    )
    return figure
