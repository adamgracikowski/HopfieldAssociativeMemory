"""Command-line interface ``hopfield-analysis``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from matplotlib.figure import Figure

from hopfield_analysis import style
from hopfield_analysis.datasets import Registry, plot_patterns
from hopfield_analysis.experiments import load_summary, plot_classes, plot_recovery
from hopfield_analysis.recall import Recall
from hopfield_analysis.runs import animate, parse_frames, plot_frames, plot_run, select_frames
from hopfield_analysis.weights import load_weights, plot_weights

OUT_HELP = "output path without extension; writes <out>.pdf and <out>.png"


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command given by ``argv``.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` if
            ``None``.

    Returns:
        The exit code: 0 on success, 1 if an input is missing or invalid
        (the reason is printed to stderr without a traceback), 2 for invalid
        arguments.
    """
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, KeyError, pl.exceptions.PolarsError) as error:
        print(f"{arguments.prog}: error: {describe(error)}", file=sys.stderr)
        return 1


def describe(error: Exception) -> str:
    """One-line description of an error caused by a missing or invalid input.

    Args:
        error: The error.

    Returns:
        For a file error the reason and the file, e.g.
        ``No such file or directory: `run/history.npy```; for a missing key of
        a JSON file the key; otherwise the first line of the message.
    """
    if isinstance(error, OSError) and error.filename is not None:
        return f"{error.strerror}: `{error.filename}`"
    if isinstance(error, KeyError):
        return f"missing field {error} in an input file"
    return str(error).splitlines()[0]


def at_least(minimum: int):
    """Argument type of integers not smaller than ``minimum``."""

    def convert(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, not {value}")
        return value

    convert.__name__ = "integer"
    return convert


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
        "--max",
        type=at_least(2),
        default=8,
        help="largest number of evenly spaced frames, at least 2 (default 8)",
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
    gif.add_argument(
        "--scale", type=at_least(1), help="image pixels per bitmap pixel (default: auto)"
    )
    gif.add_argument(
        "--duration", type=at_least(1), default=300, help="frame time in ms (default 300)"
    )
    gif.add_argument(
        "--hold", type=at_least(1), default=1500, help="last frame time in ms (default 1500)"
    )
    gif.add_argument(
        "--max-frames",
        type=at_least(2),
        default=200,
        help="largest number of frames, at least 2 (default 200)",
    )
    gif.set_defaults(handler=_animate)

    view = commands.add_parser("view", help="step through the frames of a recall run in a window")
    view.add_argument("run", type=Path, help="output directory of `compute recall`")
    view.set_defaults(handler=_view)

    patterns = commands.add_parser("patterns", help="bitmaps of the patterns of a dataset")
    patterns.add_argument("name", help="dataset name in the registry")
    patterns.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    patterns.add_argument(
        "--registry",
        type=Path,
        help="registry file (default: data/datasets/datasets.toml found from the current "
        "directory upwards)",
    )
    patterns.add_argument(
        "--columns",
        type=at_least(1),
        default=6,
        help="largest number of panels in a row (default 6)",
    )
    patterns.set_defaults(handler=_patterns)

    weights = commands.add_parser("weights", help="heat map of the weights of a train run")
    weights.add_argument("train", type=Path, help="output directory of `compute train`")
    weights.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    weights.set_defaults(handler=_weights)

    recovery = commands.add_parser(
        "recovery",
        help="recovered fraction against the noise level of an experiment",
    )
    recovery.add_argument("results", type=Path, help="results directory of an experiment")
    recovery.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    recovery.set_defaults(handler=_recovery)

    classes = commands.add_parser(
        "classes", help="fractions of the final state classes of an experiment"
    )
    classes.add_argument("results", type=Path, help="results directory of an experiment")
    classes.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    classes.set_defaults(handler=_classes)
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


def _view(args: argparse.Namespace) -> int:
    """Opens the viewer in a window.

    pyplot, and with it a GUI backend, is imported only here. The keys of the
    viewer are taken from the toolbar shortcuts while the window is open, so
    stepping through frames does not also undo a zoom. The viewer is
    referenced until the window closes, because matplotlib keeps the
    callbacks of its widgets only through weak references.
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    from hopfield_analysis.viewer import RunViewer, free_keymaps

    recall = Recall.load(args.run)
    style.apply()
    with mpl.rc_context(free_keymaps()):
        figure = plt.figure(figsize=(8, 8), layout="constrained")
        figure.canvas.manager.set_window_title(f"{recall.dataset}: {args.run}")
        viewer = RunViewer(figure, recall)
        plt.show()
    del viewer
    return 0


def _patterns(args: argparse.Namespace) -> int:
    if args.registry is None:
        registry = Registry.discover(Path.cwd())
    else:
        registry = Registry.from_file(args.registry)
    figure = plot_patterns(registry.load(args.name), columns=args.columns)
    return _save(figure, args.out)


def _weights(args: argparse.Namespace) -> int:
    return _save(plot_weights(load_weights(args.train)), args.out)


def _recovery(args: argparse.Namespace) -> int:
    return _save(plot_recovery(load_summary(args.results)), args.out)


def _classes(args: argparse.Namespace) -> int:
    return _save(plot_classes(load_summary(args.results)), args.out)
