import dataclasses

import pytest
from PIL import Image, ImageSequence

from hopfield_analysis.runs import animate, parse_frames, plot_frames, plot_run, select_frames


def test_select_frames_keeps_all_or_spaces_them_evenly():
    assert select_frames(5, 8) == [0, 1, 2, 3, 4]
    assert select_frames(10, 8) == [0, 1, 3, 4, 5, 6, 8, 9]
    assert select_frames(100, 3) == [0, 50, 99]
    for count in range(2, 60):
        for maximum in range(2, 12):
            frames = select_frames(count, maximum)
            assert frames[0] == 0 and frames[-1] == count - 1
            assert len(frames) == min(count, maximum) == len(set(frames))
            assert frames == sorted(frames)
    with pytest.raises(ValueError, match="at least 2 frames"):
        select_frames(10, 1)


def test_parse_frames_checks_the_range():
    assert parse_frames("0,5,2", 10) == [0, 5, 2]
    with pytest.raises(ValueError, match="frame 10 does not exist"):
        parse_frames("0,10", 10)
    with pytest.raises(ValueError):
        parse_frames("0,a", 10)


def test_plot_frames_shows_a_captioned_panel_per_frame(recall):
    figure = plot_frames(recall, [0, 4, 9], columns=2)
    visible = [ax for ax in figure.axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible] == ["klatka 0", "klatka 4", "klatka 9"]
    assert visible[-1].get_xlabel() == "$E$ = −66,67"
    assert len(figure.axes) == 4


def test_plot_frames_rounds_large_energies(recall):
    large = dataclasses.replace(recall, energy=recall.energy * 1000)
    figure = plot_frames(large, [9])
    assert figure.axes[0].get_xlabel() == "$E$ = −66667"


def test_plot_run_highlights_the_target(recall):
    figure = plot_run(recall)
    energy_ax, overlap_ax = figure.axes
    assert len(energy_ax.get_lines()) == 1
    assert len(overlap_ax.get_lines()) == 6
    [legend] = figure.legends
    assert [text.get_text() for text in legend.get_texts()] == [
        "wzorzec 4 (cel)",
        "pozostałe wzorce",
    ]


def test_animate_writes_a_looping_gif(recall, tmp_path):
    out = animate(recall, tmp_path / "gif" / "run.gif", duration=100, hold=900, max_frames=4)
    with Image.open(out) as image:
        assert image.size == (14 * 18, 9 * 18)
        assert image.info["loop"] == 0
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(image)]
    assert durations == [100, 100, 100, 900]
    small = animate(recall, tmp_path / "small.gif", scale=2)
    with Image.open(small) as image:
        assert image.size == (28, 18)
        assert image.n_frames == 10
