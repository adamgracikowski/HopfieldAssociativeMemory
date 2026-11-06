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
    with pytest.raises(ValueError, match="frame 12 does not exist"):
        main(["frames", RUN, "--out", str(tmp_path / "f"), "--frames", "12"])
    with pytest.raises(SystemExit):
        main(["frames", RUN, "--out", str(tmp_path / "f"), "--frames", "0", "--max", "3"])


def test_animate_writes_a_gif(tmp_path, capsys):
    out = tmp_path / "run.gif"
    assert main(["animate", RUN, "--out", str(out), "--max-frames", "3"]) == 0
    assert out.read_bytes()[:6] == b"GIF89a"
    assert "written to" in capsys.readouterr().out
