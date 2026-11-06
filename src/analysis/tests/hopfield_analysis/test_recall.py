import json
import shutil

import numpy as np
import pytest

from hopfield_analysis.recall import Recall

from .conftest import FIXTURES


def test_loads_the_files_written_by_compute(recall):
    assert recall.frames.dtype == np.int8
    assert recall.frames.shape == (10, 7, 7)
    assert set(np.unique(recall.frames)) == {-1, 1}
    assert recall.count == len(recall.energy) == len(recall.changes) == 10
    assert recall.overlaps.shape == (10, 5)
    assert recall.metadata["outcome"]["frames"] == recall.count
    assert (recall.dataset, recall.mode, recall.learning) == ("small-7x7", "asynchronous", "hebb")
    assert (recall.target, recall.nearest) == (2, 2)


def test_frames_and_statistics_agree(recall):
    """The last frame is the target pattern (overlap 1), the first one has 14
    of 49 pixels flipped, so its overlap is ``(49 - 2 * 14) / 49 = 21 / 49``.
    Energies never increase in asynchronous mode."""
    assert recall.overlaps[-1, 2] == 1.0
    assert recall.overlaps[0, 2] == pytest.approx(21 / 49)
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
