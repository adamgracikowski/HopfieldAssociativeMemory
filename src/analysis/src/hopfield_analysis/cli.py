"""Command-line interface ``hopfield-analysis``."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure

from hopfield_analysis import style
from hopfield_analysis.experiments import load_summary, plot_recovery
from hopfield_analysis.recall import Recall
from hopfield_analysis.runs import animate, parse_frames, plot_frames, plot_run, select_frames

OUT_HELP = "output path without extension; writes <out>.pdf and <out>.png"


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command given by ``argv``.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` if
            ``None``.

    Returns:
        The exit code.
    """
    args = parser().parse_args(argv)
    return args.handler(args)


def parser() -> argparse.ArgumentParser:
    """The argument parser with every command."""
    parser = argparse.ArgumentParser(
        prog="hopfield-analysis",
        description="Figures of the results written by the compute layer.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    frames = commands.add_parser("frames", help="grid of the frames of a recall run")
    frames.add_argument("run", type=Path, help="output directory of `compute recall`")
    frames.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    choice = frames.add_mutually_exclusive_group()
    choice.add_argument(
        "--max", type=int, default=8, help="largest number of evenly spaced frames (default 8)"
    )
    choice.add_argument("--frames", help="comma-separated frame numbers, e.g. 0,2,5")
    frames.set_defaults(handler=_frames)

    run = commands.add_parser("run", help="energy and overlaps of a recall run")
    run.add_argument("run", type=Path, help="output directory of `compute recall`")
    run.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    run.set_defaults(handler=_run)

    gif = commands.add_parser("animate", help="GIF animation of a recall run")
    gif.add_argument("run", type=Path, help="output directory of `compute recall`")
    gif.add_argument("--out", type=Path, required=True, help="output GIF file")
    gif.add_argument("--scale", type=int, help="image pixels per bitmap pixel (default: auto)")
    gif.add_argument("--duration", type=int, default=300, help="frame time in ms (default 300)")
    gif.add_argument("--hold", type=int, default=1500, help="last frame time in ms (default 1500)")
    gif.add_argument(
        "--max-frames", type=int, default=200, help="largest number of frames (default 200)"
    )
    gif.set_defaults(handler=_animate)

    recovery = commands.add_parser(
        "recovery",
        help="recovered fraction against the noise level of an experiment",
    )
    recovery.add_argument("results", type=Path, help="results directory of an experiment")
    recovery.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    recovery.set_defaults(handler=_recovery)
    return parser


def _save(figure: Figure, out: Path) -> int:
    for path in style.save(figure, out):
        print(f"written to {path}")
    return 0


def _frames(args: argparse.Namespace) -> int:
    recall = Recall.load(args.run)
    if args.frames is None:
        frames = select_frames(recall.count, args.max)
    else:
        frames = parse_frames(args.frames, recall.count)
    return _save(plot_frames(recall, frames), args.out)


def _run(args: argparse.Namespace) -> int:
    return _save(plot_run(Recall.load(args.run)), args.out)


def _animate(args: argparse.Namespace) -> int:
    path = animate(
        Recall.load(args.run),
        args.out,
        scale=args.scale,
        duration=args.duration,
        hold=args.hold,
        max_frames=args.max_frames,
    )
    print(f"written to {path}")
    return 0


def _recovery(args: argparse.Namespace) -> int:
    return _save(plot_recovery(load_summary(args.results)), args.out)
