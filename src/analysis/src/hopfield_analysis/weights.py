"""Weights written by ``compute train``."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from hopfield_analysis import labels, style

WEIGHTS_FILE = "weights.npy"
COEFFICIENTS_FILE = "coefficients.npy"


def load_weights(train: Path) -> np.ndarray:
    """Reads the dense ``N x N`` weight matrix from the directory of a ``compute train`` run.

    Raises:
        ValueError: If the run stored factored weights (``coefficients.npy``),
            whose full matrix is not written.
        FileNotFoundError: If the directory has no weight file.
    """
    if (train / WEIGHTS_FILE).is_file():
        return np.load(train / WEIGHTS_FILE)
    if (train / COEFFICIENTS_FILE).is_file():
        raise ValueError(
            f"`{train}` holds factored weights ({COEFFICIENTS_FILE}); "
            'train with `backend = "dense"` to plot the weight matrix'
        )
    raise FileNotFoundError(None, "no weight file", str(train / WEIGHTS_FILE))


def plot_weights(weights: np.ndarray) -> Figure:
    """Shows an ``N x N`` weight matrix as a heat map with a symmetric color scale.

    Negative weights are blue, positive red, zero white, with a color bar.
    """
    figure = style.new_figure(width=0.6 * style.TEXT_WIDTH, height=0.5 * style.TEXT_WIDTH)
    ax = figure.subplots()
    limit = float(np.abs(weights).max()) or 1.0
    image = ax.imshow(weights, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="auto")
    ax.set_xlabel(f"{labels.NEURON} $j$")
    ax.set_ylabel(f"{labels.NEURON} $i$")
    ax.grid(False)
    figure.colorbar(image, ax=ax, label=labels.WEIGHT)
    return figure
