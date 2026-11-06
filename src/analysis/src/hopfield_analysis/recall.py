"""Artifacts of ``compute recall``."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HISTORY_FILE = "history.npy"
FRAMES_FILE = "frames.json"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class Recall:
    """A recall run: every recorded state with its statistics.

    Attributes:
        frames: States as bitmaps, shape ``(frames, height, width)``, values
            ``1`` and ``-1``.
        energy: Energy of each frame.
        changes: Neurons changed since the previous frame.
        overlaps: Overlap of each frame with every stored pattern, shape
            ``(frames, p)``.
        metadata: The parsed ``metadata.json``.
    """

    frames: np.ndarray
    energy: np.ndarray
    changes: np.ndarray
    overlaps: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, run: Path) -> "Recall":
        """Reads the directory written by ``compute recall``.

        Args:
            run: Output directory of the run.

        Returns:
            The run.

        Raises:
            ValueError: If the files disagree on the number of frames.
        """
        frames = np.load(run / HISTORY_FILE)
        statistics = json.loads((run / FRAMES_FILE).read_text(encoding="utf-8"))
        metadata = json.loads((run / METADATA_FILE).read_text(encoding="utf-8"))
        recall = cls(
            frames=frames,
            energy=np.asarray(statistics["energy"], dtype=float),
            changes=np.asarray(statistics["changes"], dtype=int),
            overlaps=np.asarray(statistics["overlaps"], dtype=float),
            metadata=metadata,
        )
        lengths = {len(recall.frames), len(recall.energy), len(recall.changes)}
        lengths.add(len(recall.overlaps))
        if len(lengths) != 1:
            raise ValueError(f"`{run}` has inconsistent numbers of frames: {sorted(lengths)}")
        return recall

    @property
    def count(self) -> int:
        """Number of frames."""
        return len(self.frames)

    @property
    def dataset(self) -> str:
        """Name of the dataset."""
        return self.metadata["dataset"]["name"]

    @property
    def mode(self) -> str:
        """Update mode, ``synchronous`` or ``asynchronous``."""
        return self.metadata["config"]["dynamics"]["mode"]

    @property
    def learning(self) -> str:
        """Name of the learning rule as ``compute`` prints it, e.g. ``oja (subspace)``."""
        learning = self.metadata["config"]["learning"]
        if learning["rule"] == "oja":
            return f"oja ({learning['variant']})"
        return learning["rule"]

    @property
    def target(self) -> int | None:
        """Stored pattern the initial state was made from, if any."""
        return self.metadata["outcome"]["target"]

    @property
    def nearest(self) -> int:
        """Stored pattern with the largest absolute overlap with the final state."""
        return self.metadata["outcome"]["nearest"]["pattern"]
