import pytest

from hopfield_analysis.cli import main

from .conftest import FIXTURES

RUN = str(FIXTURES / "recall")


@pytest.mark.parametrize("command", ["frames", "run"])
def test_figure_commands_write_a_pdf_and_a_png(command, tmp_path, capsys):
    out = tmp_path / "plots" / command
    assert main([command, RUN, "--out", str(out)]) == 0
    assert (tmp_path / "plots" / f"{command}.pdf").read_bytes()[:5] == b"%PDF-"
    assert (tmp_path / "plots" / f"{command}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert capsys.readouterr().out.count("written to") == 2


def test_frames_accepts_an_explicit_choice(tmp_path):
    assert main(["frames", RUN, "--out", str(tmp_path / "f"), "--frames", "0,9"]) == 0
    assert main(["frames", RUN, "--out", str(tmp_path / "f"), "--frames", "12"]) == 1
    with pytest.raises(SystemExit):
        main(["frames", RUN, "--out", str(tmp_path / "f"), "--frames", "0", "--max", "3"])


def test_animate_writes_a_gif(tmp_path, capsys):
    out = tmp_path / "run.gif"
    assert main(["animate", RUN, "--out", str(out), "--max-frames", "3"]) == 0
    assert out.read_bytes()[:6] == b"GIF89a"
    assert "written to" in capsys.readouterr().out


def test_view_opens_the_viewer_in_a_window(monkeypatch):
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    assert main(["view", RUN]) == 0
    [figure] = shown
    assert len(figure.axes) == 4
    assert figure.axes[0].get_title().startswith("klatka 0/9")
    plt.close("all")


def test_reports_missing_or_invalid_inputs_without_a_traceback(tmp_path, capsys):
    assert main(["frames", str(tmp_path / "missing"), "--out", str(tmp_path / "f")]) == 1
    error = capsys.readouterr().err
    assert error.startswith("hopfield-analysis: error: `")
    assert "history.npy` not found" in error
    assert main(["frames", RUN, "--out", str(tmp_path / "f"), "--frames", "12"]) == 1
    assert "frame 12 does not exist, the run has 10 frames" in capsys.readouterr().err


def test_patterns_and_weights_write_figures(tmp_path):
    registry = str(FIXTURES.parents[3] / "data" / "datasets" / "datasets.toml")
    out = tmp_path / "patterns"
    assert main(["patterns", "small-7x7", "--out", str(out), "--registry", registry]) == 0
    assert out.with_suffix(".pdf").is_file() and out.with_suffix(".png").is_file()
    out = tmp_path / "weights"
    assert main(["weights", str(FIXTURES / "train"), "--out", str(out)]) == 0
    assert out.with_suffix(".png").is_file()
    assert main(["weights", str(tmp_path), "--out", str(out)]) == 1
