import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent

from hopfield_analysis import style
from hopfield_analysis.viewer import RunViewer


@pytest.fixture
def viewer(recall):
    return RunViewer(style.new_figure(width=6, height=6), recall)


def press(viewer, key):
    event = KeyEvent("key_press_event", viewer.figure.canvas, key)
    viewer.figure.canvas.callbacks.process("key_press_event", event)


def shown(viewer):
    return np.asarray(viewer._image.get_array())


def test_starts_at_the_first_frame(viewer, recall):
    assert viewer.frame == 0
    assert np.array_equal(shown(viewer), recall.frames[0])
    assert viewer.title == "klatka 0/9, $E$ = \u22124,41, zmiany: 0"


def test_set_frame_updates_bitmap_title_markers_and_slider(viewer, recall):
    viewer.set_frame(4)
    assert np.array_equal(shown(viewer), recall.frames[4])
    assert viewer._bitmap_ax.get_title().startswith("klatka 4/9")
    assert all(list(marker.get_xdata()) == [4, 4] for marker in viewer._markers)
    assert viewer.slider.val == 4
    viewer.set_frame(50)
    assert viewer.frame == 9
    viewer.set_frame(-3)
    assert viewer.frame == 0


def test_keys_move_by_one_frame_and_jump_to_the_ends(viewer):
    press(viewer, "right")
    press(viewer, "right")
    assert viewer.frame == 2
    press(viewer, "left")
    assert viewer.frame == 1
    press(viewer, "end")
    assert viewer.frame == 9
    press(viewer, "right")
    assert viewer.frame == 9
    press(viewer, "home")
    press(viewer, "left")
    assert viewer.frame == 0
    press(viewer, "x")
    assert viewer.frame == 0


def test_the_slider_reaches_every_frame(viewer, recall):
    for frame in range(recall.count):
        viewer.slider.set_val(frame)
        assert viewer.frame == frame
        assert np.array_equal(shown(viewer), recall.frames[frame])
