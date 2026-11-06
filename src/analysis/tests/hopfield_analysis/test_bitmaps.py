import numpy as np

from hopfield_analysis import style
from hopfield_analysis.bitmaps import draw_bitmap, to_image

STATE = np.array([[1, -1, -1], [-1, 1, 1]], dtype=np.int8)


def test_to_image_draws_ones_black_and_enlarges():
    image = to_image(STATE, scale=3)
    assert image.mode == "L"
    assert image.size == (9, 6)
    pixels = np.asarray(image)
    assert pixels[0, 0] == 0 and pixels[2, 2] == 0
    assert pixels[0, 3] == 255
    assert np.array_equal(pixels[::3, ::3], np.where(STATE == 1, 0, 255))


def test_draw_bitmap_maps_one_to_black_without_ticks():
    figure = style.new_figure()
    ax = figure.subplots()
    image = draw_bitmap(ax, STATE)
    colors = image.to_rgba(STATE.astype(float))
    assert np.allclose(colors[0, 0, :3], 0)
    assert np.allclose(colors[0, 1, :3], 1)
    assert image.get_interpolation() == "nearest"
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []
