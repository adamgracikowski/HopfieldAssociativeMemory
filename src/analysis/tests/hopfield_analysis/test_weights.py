from pathlib import Path

import numpy as np
import pytest

from hopfield_analysis.datasets import Registry
from hopfield_analysis.weights import load_weights, plot_weights

from .conftest import FIXTURES


def test_loads_the_hebbian_weights_written_by_compute():
    """The fixture holds the Hebbian weights of ``small-7x7``:
    ``w_ij = sum_mu xi_i^mu xi_j^mu / N`` for ``i != j`` and ``w_ii = 0``,
    computed here directly from the patterns of the registry."""
    weights = load_weights(FIXTURES / "train")
    assert weights.shape == (49, 49)
    patterns = Registry.discover(Path(__file__).parent).load("small-7x7").reshape(5, 49)
    expected = patterns.T.astype(float) @ patterns / 49
    np.fill_diagonal(expected, 0)
    assert np.allclose(weights, expected)
    assert np.array_equal(weights, weights.T)


def test_rejects_factored_weights_and_missing_files(tmp_path):
    np.save(tmp_path / "coefficients.npy", np.zeros((4, 2)))
    with pytest.raises(ValueError, match='backend = "dense"'):
        load_weights(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_weights(tmp_path / "missing")


def test_plot_weights_uses_a_symmetric_scale():
    weights = np.array([[0.0, 0.2], [-0.5, 0.0]])
    figure = plot_weights(weights)
    image = figure.axes[0].images[0]
    assert image.get_clim() == (-0.5, 0.5)
    assert len(figure.axes) == 2
