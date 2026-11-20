"""Presentation layer of the Hopfield associative memory project.

The package reads the files written by the compute layer (``src/compute``) and
visualizes them; it does not compute any results itself. Plot functions return a
``matplotlib.figure.Figure`` in the report style with Polish labels.
"""

from importlib.metadata import version

__version__ = version("hopfield-analysis")
