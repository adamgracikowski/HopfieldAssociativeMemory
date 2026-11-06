"""Interactive viewer of the frames of a recall run."""

from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider

from hopfield_analysis import labels, style
from hopfield_analysis.bitmaps import draw_bitmap
from hopfield_analysis.recall import Recall

KEYS = {"right": 1, "left": -1}
"""Keys that move by a frame; ``home`` and ``end`` jump to the first and last."""


class RunViewer:
    """Steps through the frames of a recall run in a figure.

    The figure shows the current frame as a bitmap, below it the energy and
    the overlaps with every pattern against the frame with a vertical marker
    at the current frame, and a slider. The arrow keys move by one frame,
    ``home`` and ``end`` jump to the first and the last frame.

    The viewer works on any figure: ``hopfield-analysis view`` opens it in a
    window, the tests drive it on a figure without a GUI.

    Attributes:
        recall: The run.
        frame: Number of the current frame.
        slider: The frame slider.
    """

    def __init__(self, figure: Figure, recall: Recall) -> None:
        """Draws the first frame of ``recall`` into ``figure``.

        Args:
            figure: An empty figure.
            recall: The run.
        """
        self.figure = figure
        self.recall = recall
        self.frame = 0
        grid = figure.add_gridspec(3, 2, height_ratios=[3, 1.4, 0.15])
        self._bitmap_ax = figure.add_subplot(grid[0, :])
        energy_ax = figure.add_subplot(grid[1, 0])
        overlap_ax = figure.add_subplot(grid[1, 1])
        slider_ax = figure.add_subplot(grid[2, :])

        self._image = draw_bitmap(self._bitmap_ax, recall.frames[0])
        steps = range(recall.count)
        energy_ax.plot(steps, recall.energy, marker="o", markersize=2)
        energy_ax.set_ylabel(labels.ENERGY)
        for mu in range(recall.overlaps.shape[1]):
            highlighted = mu in (recall.target, recall.nearest)
            overlap_ax.plot(
                steps,
                recall.overlaps[:, mu],
                color=None if highlighted else "0.6",
                linewidth=1.5 if highlighted else 0.8,
            )
        overlap_ax.set_ylabel(labels.OVERLAP)
        overlap_ax.set_ylim(-1.05, 1.05)
        self._markers = []
        for ax in (energy_ax, overlap_ax):
            ax.set_xlabel(labels.FRAME)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            self._markers.append(ax.axvline(0, color="C3", linewidth=1))
        style.finish(figure)

        self.slider = Slider(
            slider_ax,
            labels.FRAME,
            0,
            max(recall.count - 1, 1),
            valinit=0,
            valstep=1,
            valfmt="%d",
        )
        self.slider.on_changed(lambda value: self.set_frame(int(value)))
        figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.set_frame(0)

    @property
    def title(self) -> str:
        """Title of the current frame: number, energy and changed neurons."""
        k = self.frame
        return (
            f"{labels.FRAME} {k}/{self.recall.count - 1}, "
            f"$E$ = {labels.number(self.recall.energy[k])}, "
            f"{labels.CHANGES}: {self.recall.changes[k]}"
        )

    def set_frame(self, frame: int) -> None:
        """Shows ``frame``, clamped to the frames of the run.

        Args:
            frame: Frame number.
        """
        self.frame = min(max(frame, 0), self.recall.count - 1)
        self._image.set_data(self.recall.frames[self.frame])
        self._bitmap_ax.set_title(self.title)
        for marker in self._markers:
            marker.set_xdata([self.frame, self.frame])
        if int(self.slider.val) != self.frame:
            self.slider.set_val(self.frame)
        self.figure.canvas.draw_idle()

    def _on_key(self, event: KeyEvent) -> None:
        if event.key in KEYS:
            self.set_frame(self.frame + KEYS[event.key])
        elif event.key == "home":
            self.set_frame(0)
        elif event.key == "end":
            self.set_frame(self.recall.count - 1)
