"""Results of ``compute experiment``: loading and plotting.

An experiment directory holds ``trials.parquet`` (one row per run),
``summary.json`` (aggregates of every grid cell) and ``metadata.json``. All
numbers are computed by the compute layer; this module only reads and plots
them.
"""

import json
from pathlib import Path
from typing import Any

import polars as pl
from matplotlib.figure import Figure

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

    One panel per dataset, one line per learning rule and update mode, with
    the 95% Wilson intervals from the summary as error bars.

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
    figure = Figure(figsize=(4.5 * len(datasets), 3.5), layout="constrained")
    axes = figure.subplots(1, len(datasets), squeeze=False, sharey=True)[0]
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
                capsize=3,
                label=f"{learning}, {mode}",
            )
        ax.set_title(dataset)
        ax.set_xlabel("noise (fraction of flipped pixels)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("recovered fraction")
    axes[-1].legend(fontsize="small")
    return figure
