"""Figures and animations of a recall run."""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from hopfield_analysis import labels, style
from hopfield_analysis.bitmaps import draw_bitmap, to_image
from hopfield_analysis.recall import Recall

GIF_SIDE = 256
"""Default longer side of an animation in pixels, reached with an integer scale."""
ENERGY_WITHOUT_DECIMALS = 1000
"""Smallest absolute energy shown in a frame caption without decimals."""


def select_frames(count: int, maximum: int) -> list[int]:
    """Chooses at most ``maximum`` of ``count`` frames, evenly spaced.

    Args:
        count: Number of frames.
        maximum: Largest number of chosen frames, at least 2.

    Returns:
        All frames if there are at most ``maximum``, otherwise ``maximum``
        evenly spaced frames that always include the first and the last.

    Raises:
        ValueError: If ``maximum`` is smaller than 2.
    """
    if maximum < 2:
        raise ValueError(f"at least 2 frames are needed, not {maximum}")
    if count <= maximum:
        return list(range(count))
    return [round(k * (count - 1) / (maximum - 1)) for k in range(maximum)]


def parse_frames(text: str, count: int) -> list[int]:
    """Parses comma-separated frame numbers such as ``0,2,5`` of a run with ``count`` frames.

    Returns:
        The frame numbers in the given order.

    Raises:
        ValueError: If an entry is not an integer or not a frame of the run.
    """
    frames = [int(part) for part in text.split(",")]
    for frame in frames:
        if not 0 <= frame < count:
            raise ValueError(f"frame {frame} does not exist, the run has {count} frames")
    return frames


def plot_frames(
    recall: Recall, frames: Sequence[int], columns: int = 8, width: float = style.TEXT_WIDTH
) -> Figure:
    """Shows the chosen frames as bitmaps in a grid.

    Every panel has the frame number above and its energy below, with two
    decimals, or none from 1000 on, so the label of a large network fits
    under its panel.

    Args:
        recall: The run.
        frames: Frame numbers to show, in order.
        columns: Largest number of panels in a row.
        width: Figure width in inches; a narrower one keeps the report font size
            (see :func:`hopfield_analysis.datasets.plot_patterns`).
    """
    columns = min(columns, len(frames))
    rows = math.ceil(len(frames) / columns)
    bitmap_height, bitmap_width = recall.frames.shape[1:]
    panel = width / columns
    figure = style.new_figure(
        width=width, height=rows * (panel * bitmap_height / bitmap_width + 0.45)
    )
    axes = figure.subplots(rows, columns, squeeze=False)
    for ax in axes.flat[len(frames) :]:
        ax.set_visible(False)
    for ax, frame in zip(axes.flat, frames, strict=False):
        draw_bitmap(ax, recall.frames[frame])
        ax.set_title(f"{labels.FRAME} {frame}")
        energy = recall.energy[frame]
        digits = 2 if abs(energy) < ENERGY_WITHOUT_DECIMALS else 0
        ax.set_xlabel(f"$E$ = {labels.number(energy, digits)}")
    return figure


def plot_run(recall: Recall) -> Figure:
    """Plots the energy and the overlaps with every pattern against the frame.

    The target pattern (or, without a target, the pattern nearest to the
    final state) is drawn in color, the other patterns as thin gray lines.
    """
    figure = style.new_figure(height=2.6)
    energy_ax, overlap_ax = figure.subplots(1, 2)
    steps = np.arange(recall.count)
    energy_ax.plot(steps, recall.energy, marker="o", markersize=3)
    energy_ax.set_ylabel(labels.ENERGY)
    highlighted = recall.target if recall.target is not None else recall.nearest
    role = labels.TARGET if recall.target is not None else labels.NEAREST
    others = [mu for mu in range(recall.overlaps.shape[1]) if mu != highlighted]
    for k, mu in enumerate(others):
        overlap_ax.plot(
            steps,
            recall.overlaps[:, mu],
            color="0.6",
            linewidth=0.8,
            label=labels.OTHER_PATTERNS if k == 0 else None,
        )
    overlap_ax.plot(
        steps,
        recall.overlaps[:, highlighted],
        color="C0",
        marker="o",
        markersize=3,
        linewidth=1.8,
        label=f"{labels.PATTERN} {highlighted} ({role})",
    )
    overlap_ax.set_ylabel(labels.OVERLAP)
    overlap_ax.set_ylim(-1.05, 1.05)
    for ax in (energy_ax, overlap_ax):
        ax.set_xlabel(labels.FRAME)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, names = overlap_ax.get_legend_handles_labels()
    figure.legend(handles[::-1], names[::-1], loc="outside lower center", ncols=2)
    return figure


def animate(
    recall: Recall,
    out: Path,
    scale: int | None = None,
    duration: int = 300,
    hold: int = 1500,
    max_frames: int = 200,
) -> Path:
    """Writes the run as a looping GIF of enlarged bitmaps.

    Args:
        recall: The run.
        out: Output file, usually ``.gif``; missing directories are created.
        scale: Side of the square of image pixels for one bitmap pixel; by
            default the largest integer that keeps the longer side at most
            256 pixels, and at least 1.
        duration: Display time of a frame in milliseconds.
        hold: Display time of the last frame in milliseconds.
        max_frames: Largest number of frames, chosen with :func:`select_frames`.

    Returns:
        The written file (identical consecutive frames are merged).
    """
    if scale is None:
        scale = max(1, GIF_SIDE // max(recall.frames.shape[1:]))
    frames = select_frames(recall.count, max_frames)
    images = [to_image(recall.frames[frame], scale) for frame in frames]
    durations = [duration] * (len(images) - 1) + [hold]
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )
    return out
