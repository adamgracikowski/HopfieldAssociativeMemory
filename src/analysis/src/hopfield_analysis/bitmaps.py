"""States of the network as rectangular bitmaps: ``1`` black, ``-1`` white."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from PIL import Image


def draw_bitmap(ax: Axes, state: np.ndarray) -> AxesImage:
    """Draws a bitmap without ticks or grid.

    Uses interpolation ``auto`` (see :func:`hopfield_analysis.style.save`).

    Args:
        ax: Target axes.
        state: Bitmap of shape ``(height, width)`` with values ``1`` and ``-1``.

    Returns:
        The image, whose data can be replaced with ``set_data``.
    """
    image = ax.imshow(state, cmap="gray_r", vmin=-1, vmax=1, interpolation="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return image


def to_image(state: np.ndarray, scale: int = 1) -> Image.Image:
    """Converts a bitmap to a grayscale image enlarged ``scale`` times.

    Args:
        state: Bitmap of shape ``(height, width)`` with values ``1`` and ``-1``.
        scale: Side of the square of image pixels for one bitmap pixel.

    Returns:
        A mode ``L`` image of size ``(width * scale, height * scale)``, black
        for ``1`` and white for ``-1``.
    """
    pixels = np.where(state == 1, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    height, width = state.shape
    return image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
