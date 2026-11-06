import json
import shutil

import numpy as np
import pytest

from hopfield_analysis.datasets import Registry
from hopfield_analysis.recall import Recall

from .conftest import FIXTURES


def test_loads_the_files_written_by_compute(recall):
    assert recall.frames.dtype == np.int8
    assert recall.frames.shape == (10, 9, 14)
    assert set(np.unique(recall.frames)) == {-1, 1}
    assert recall.count == len(recall.energy) == len(recall.changes) == 10
    assert recall.overlaps.shape == (10, 6)
    assert recall.metadata["outcome"]["frames"] == recall.count
    assert (recall.dataset, recall.mode, recall.learning) == (
        "animals-14x9",
        "asynchronous",
        "hebb",
    )
    assert (recall.target, recall.nearest) == (4, 4)


def test_frames_and_statistics_agree(recall):
    """The last frame is the target pattern (overlap 1), the first one has 30
    of 126 pixels flipped, so its overlap is ``(126 - 2 * 30) / 126 = 66 / 126``.
    Energies never increase in asynchronous mode."""
    assert recall.overlaps[-1, 4] == 1.0
    assert recall.overlaps[0, 4] == pytest.approx(66 / 126)
    assert np.all(np.diff(recall.energy) <= 0)
    assert recall.changes[0] == 0
    assert recall.changes.sum() == recall.metadata["outcome"]["changes"]


def test_rejects_inconsistent_files(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIXTURES / "recall", run)
    frames = json.loads((run / "frames.json").read_text(encoding="utf-8"))
    frames["energy"].pop()
    (run / "frames.json").write_text(json.dumps(frames), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent numbers of frames"):
        Recall.load(run)


def test_names_oja_with_its_variant(recall):
    metadata = json.loads(json.dumps(recall.metadata))
    metadata["config"]["learning"] = {"rule": "oja", "variant": "subspace", "eta": 0.5}
    oja = Recall(recall.frames, recall.energy, recall.changes, recall.overlaps, metadata)
    assert oja.learning == "oja (subspace)"


def test_frames_are_rows_of_the_bitmap(recall):
    """Overlaps recomputed from the frames and the patterns of the registry,
    ``m = sum_i x_i xi_i / N`` on the flattened bitmaps, equal the overlaps
    written by ``compute``; with height and width swapped they would not."""
    patterns = Registry.discover(FIXTURES).load("animals-14x9").reshape(6, -1)
    frames = recall.frames.reshape(recall.count, -1)
    assert np.allclose(frames @ patterns.T / 126, recall.overlaps)
    swapped = recall.frames.transpose(0, 2, 1).reshape(recall.count, -1)
    assert not np.allclose(swapped @ patterns.T / 126, recall.overlaps)
