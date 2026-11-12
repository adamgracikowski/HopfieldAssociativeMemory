"""Results of ``compute benchmark``: loading and plotting.

A benchmark directory holds ``benchmark.parquet`` (one row per measurement:
set, backend, learning rule and repeat) and ``metadata.json``. The times are
wall-clock times of the machine that ran it; this module only aggregates and
plots them.
"""

from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from hopfield_analysis import labels, style

BENCHMARK_FILE = "benchmark.parquet"
RANDOM_SOURCE = "random"
"""Source name of random pattern sets."""
MIB = 2**20
KEYS = ["source", "neurons", "patterns", "backend", "learning"]
DATASET_MARKERS = ["^", "v", "D", "P"]
"""Markers of the registered datasets, in the order of the table."""


def load_benchmark(results: Path) -> pl.DataFrame:
    """Reads the table of a benchmark.

    Args:
        results: Directory written by ``compute benchmark``.

    Returns:
        One row per measurement with the columns described in
        ``src/compute/README.md`` (``source``, ``neurons``, ``patterns``,
        ``backend``, ``learning``, ``repeat``, ``epochs``, ``train_seconds``,
        ``weights_bytes``, ``train_peak_bytes``, ``fields_seconds``,
        ``sync_seconds``, ``sync_steps``, ``async_seconds``,
        ``async_sweeps``, ``recall_peak_bytes``).
    """
    return pl.read_parquet(results / BENCHMARK_FILE)


def medians(table: pl.DataFrame) -> pl.DataFrame:
    """The median of every measurement over the repeats.

    The time of a sweep is the time of the asynchronous run divided by its
    sweeps plus one: the last sweep only confirms the fixed point and is not
    counted in ``async_sweeps``.

    Args:
        table: Table from :func:`load_benchmark`.

    Returns:
        One row per (source, ``N``, ``p``, backend, rule), in the order of
        the table, with the medians ``train_seconds``, ``fields_seconds``,
        ``sync_seconds``, ``sweep_seconds``, ``train_peak_bytes`` and
        ``recall_peak_bytes``, the ``weights_bytes`` and the ``epochs`` of
        the first repeat and the number of ``repeats``.
    """
    sweep = pl.col("async_seconds") / (pl.col("async_sweeps") + 1)
    return table.group_by(KEYS, maintain_order=True).agg(
        pl.len().alias("repeats"),
        pl.col("epochs").first(),
        pl.col("weights_bytes").first(),
        *(
            pl.col(name).median()
            for name in (
                "train_seconds",
                "fields_seconds",
                "sync_seconds",
                "train_peak_bytes",
                "recall_peak_bytes",
            )
        ),
        sweep.median().alias("sweep_seconds"),
    )


def plot_benchmark(table: pl.DataFrame) -> Figure:
    """Plots the time and memory of training and recall against ``N``.

    Four log-log panels: the time of training, of the local fields of one
    state (the main cost of a synchronous step), of one asynchronous sweep,
    and the memory: the bytes of the weights (solid) and the peak growth of
    the heap during training (dashed), with the size ``8 N^2`` of a dense
    matrix as a dotted reference. Every (backend, rule) of the random sets
    is a line, in the order of the random sets in the table, the registered
    datasets are markers in the color of their (backend, rule).

    Args:
        table: Table from :func:`load_benchmark`.

    Returns:
        The figure in the report style with Polish labels.

    Raises:
        ValueError: If the table is empty.
    """
    if table.is_empty():
        raise ValueError("the benchmark has no measurements")
    data = medians(table)
    data = data.sort(pl.col("source") != RANDOM_SOURCE, maintain_order=True)
    series = data.select("backend", "learning").unique(maintain_order=True).rows()
    colors = {key: f"C{k}" for k, key in enumerate(series)}
    datasets = [s for s in data["source"].unique(maintain_order=True) if s != RANDOM_SOURCE]
    markers = {name: DATASET_MARKERS[k % len(DATASET_MARKERS)] for k, name in enumerate(datasets)}
    panels = [
        ("train_seconds", labels.TRAIN_TIME),
        ("fields_seconds", labels.FIELDS_TIME),
        ("sweep_seconds", labels.SWEEP_TIME),
        ("weights_bytes", labels.MEMORY),
    ]
    figure = style.new_figure(height=4.6)
    axes = figure.subplots(2, 2, sharex=True).ravel()
    for ax, (column, ylabel) in zip(axes, panels, strict=True):
        for (backend, learning), rows in data.group_by(
            ["backend", "learning"], maintain_order=True
        ):
            color = colors[(backend, learning)]
            name = f"{labels.backend(backend)}, {labels.rule(learning)}"
            scale = MIB if column == "weights_bytes" else 1
            random = rows.filter(pl.col("source") == RANDOM_SOURCE).sort("neurons")
            if random.height:
                ax.plot(
                    random["neurons"].to_numpy(),
                    random[column].to_numpy() / scale,
                    color=color,
                    marker="o",
                    markersize=2.5,
                    linewidth=1,
                    label=name,
                )
                if column == "weights_bytes" and random["train_peak_bytes"].null_count() == 0:
                    ax.plot(
                        random["neurons"].to_numpy(),
                        random["train_peak_bytes"].to_numpy() / scale,
                        color=color,
                        linestyle="--",
                        linewidth=0.9,
                    )
            for dataset in datasets:
                points = rows.filter(pl.col("source") == dataset)
                if points.height:
                    ax.plot(
                        points["neurons"].to_numpy(),
                        points[column].to_numpy() / scale,
                        color=color,
                        marker=markers[dataset],
                        markersize=4,
                        linestyle="none",
                    )
        if column == "weights_bytes":
            sizes = np.unique(data["neurons"].to_numpy()).astype(float)
            ax.plot(
                sizes,
                8 * sizes**2 / MIB,
                color="black",
                linestyle=":",
                linewidth=0.9,
                label=labels.DENSE_SIZE,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
    figure.supxlabel(labels.NEURONS, fontsize=9)
    handles, names = [], []
    for ax in axes:
        for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
            if name not in names:
                handles.append(handle)
                names.append(name)
    for dataset in datasets:
        handles.append(Line2D([], [], color="0.3", marker=markers[dataset], linestyle="none"))
        names.append(dataset)
    if data["train_peak_bytes"].null_count() == 0:
        handles.append(Line2D([], [], color="0.3", linestyle="--", linewidth=0.9))
        names.append(labels.TRAIN_PEAK)
    figure.legend(handles, names, loc="outside upper center", ncols=2)
    return figure
