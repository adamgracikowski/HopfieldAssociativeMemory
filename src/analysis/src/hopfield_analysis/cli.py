"""Command-line interface ``hopfield-analysis``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from matplotlib.figure import Figure

from hopfield_analysis import style
from hopfield_analysis.attractors import (
    bitmap_shapes,
    load_attractors,
    plot_accessibility,
    plot_attractors,
    plot_distinct,
    plot_energies,
    plot_outcomes,
)
from hopfield_analysis.capacity import load_capacity_summary, plot_capacity
from hopfield_analysis.datasets import Registry, plot_patterns
from hopfield_analysis.experiments import (
    METRICS,
    damage,
    dataset_sizes,
    load_metadata,
    load_summary,
    load_trials,
    plot_classes,
    plot_load,
    plot_recovery,
)
from hopfield_analysis.recall import Recall
from hopfield_analysis.runs import animate, parse_frames, plot_frames, plot_run, select_frames
from hopfield_analysis.stability import (
    aligned_fields,
    load_datasets,
    load_fields,
    load_stability,
    plot_fields,
    plot_overlaps,
    plot_stable,
    plot_unstable,
    unstable_mask,
    zero_field,
)
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
        help="recovered fraction against the damage level of an experiment",
    )
    recovery.add_argument("results", type=Path, help="results directory of an experiment")
    recovery.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    recovery.add_argument(
        "--metric",
        choices=list(METRICS),
        default="recovered",
        help="runs that ended in their target, or whose final state is nearest "
        "their target (default recovered)",
    )
    recovery.set_defaults(handler=_recovery)

    load = commands.add_parser(
        "load",
        help="recovered fraction against the number of patterns of dataset prefixes",
    )
    load.add_argument("results", type=Path, help="results directory of an experiment")
    load.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    load.add_argument(
        "--levels", type=levels, help="comma-separated damage levels, e.g. 0,0.2 (default all)"
    )
    load.add_argument(
        "--mode",
        choices=["synchronous", "asynchronous"],
        default="synchronous",
        help="update mode of the drawn cells (default synchronous)",
    )
    load.add_argument(
        "--metric", choices=list(METRICS), default="recovered", help="as in `recovery`"
    )
    load.set_defaults(handler=_load)

    classes = commands.add_parser(
        "classes", help="fractions of the final state classes of an experiment"
    )
    classes.add_argument("results", type=Path, help="results directory of an experiment")
    classes.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    classes.set_defaults(handler=_classes)

    stable = commands.add_parser(
        "stable", help="fraction of stable stored patterns of one or more experiments"
    )
    stable.add_argument("results", type=Path, nargs="+", help="results directories of experiments")
    stable.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    stable.set_defaults(handler=_stable)

    fields = commands.add_parser(
        "fields", help="histograms of the aligned fields of the stored patterns"
    )
    fields.add_argument(
        "results", type=Path, help="results directory of an experiment with stability fields"
    )
    fields.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    fields.add_argument("--datasets", type=names, help="comma-separated datasets (default: all)")
    fields.add_argument("--log", action="store_true", help="logarithmic count axis")
    fields.set_defaults(handler=_fields)

    unstable = commands.add_parser(
        "unstable", help="stored patterns of a dataset with their unstable pixels"
    )
    unstable.add_argument(
        "results", type=Path, help="results directory of an experiment with stability fields"
    )
    unstable.add_argument("--dataset", required=True, help="dataset name")
    unstable.add_argument(
        "--learning", required=True, help="learning rule as in the results, e.g. hebb"
    )
    unstable.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    unstable.add_argument(
        "--registry",
        type=Path,
        help="registry file (default: data/datasets/datasets.toml found from the current "
        "directory upwards)",
    )
    unstable.add_argument(
        "--columns",
        type=at_least(1),
        default=6,
        help="largest number of panels in a row (default 6)",
    )
    unstable.set_defaults(handler=_unstable)

    overlaps = commands.add_parser(
        "overlaps", help="heat maps of the overlaps between the patterns of datasets"
    )
    overlaps.add_argument("results", type=Path, help="results directory of an experiment")
    overlaps.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    overlaps.add_argument("--datasets", type=names, help="comma-separated datasets (default: all)")
    overlaps.set_defaults(handler=_overlaps)

    capacity = commands.add_parser(
        "capacity", help="stability of the stored patterns against the load of a capacity run"
    )
    capacity.add_argument("results", type=Path, help="results directory of `compute capacity`")
    capacity.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    capacity.set_defaults(handler=_capacity)

    random_results = "results directory of an experiment with random inputs"
    with_attractors = "results directory of an experiment with `attractors = true`"
    mode_help = "update mode of the drawn cells (default asynchronous)"

    outcomes = commands.add_parser(
        "outcomes", help="final state classes of random inputs for every dataset"
    )
    outcomes.add_argument("results", type=Path, help=random_results)
    outcomes.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    outcomes.set_defaults(handler=_outcomes)

    attractors = commands.add_parser(
        "attractors", help="bitmaps of the most frequent final states of a dataset"
    )
    attractors.add_argument("results", type=Path, help=with_attractors)
    attractors.add_argument("--dataset", required=True, help="dataset label in the results")
    attractors.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    attractors.add_argument(
        "--top", type=at_least(1), default=6, help="states per rule and mode (default 6)"
    )
    attractors.set_defaults(handler=_attractors)

    reached = commands.add_parser(
        "accessibility", help="fraction of random inputs ending in every pattern and negation"
    )
    reached.add_argument("results", type=Path, help=with_attractors)
    reached.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    reached.add_argument("--datasets", type=names, help="comma-separated datasets (default: all)")
    reached.add_argument(
        "--mode", choices=["synchronous", "asynchronous"], default="asynchronous", help=mode_help
    )
    reached.set_defaults(handler=_accessibility)

    distinct = commands.add_parser(
        "distinct", help="spurious states against the number of patterns of dataset prefixes"
    )
    distinct.add_argument("results", type=Path, help=with_attractors)
    distinct.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    distinct.set_defaults(handler=_distinct)

    energies = commands.add_parser(
        "energies", help="histograms of the final energy by final state class"
    )
    energies.add_argument("results", type=Path, help="results directory of an experiment")
    energies.add_argument("--out", type=Path, required=True, help=OUT_HELP)
    energies.add_argument("--datasets", type=names, help="comma-separated datasets (default: all)")
    energies.add_argument(
        "--mode", choices=["synchronous", "asynchronous"], default="asynchronous", help=mode_help
    )
    energies.set_defaults(handler=_energies)
    return parser


def levels(text: str) -> list[float]:
    """Argument type of a comma-separated list of damage levels."""
    try:
        return [float(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid levels `{text}`") from error


def names(text: str) -> list[str]:
    """Argument type of a comma-separated list of names."""
    values = [value.strip() for value in text.split(",")]
    if not all(values):
        raise argparse.ArgumentTypeError(f"empty name in `{text}`")
    return values


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
    kind = damage(load_metadata(args.results))
    figure = plot_recovery(load_summary(args.results), metric=args.metric, damage=kind)
    return _save(figure, args.out)


def _load(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.results)
    figure = plot_load(
        load_summary(args.results),
        dataset_sizes(metadata),
        levels=args.levels,
        mode=args.mode,
        metric=args.metric,
        damage=damage(metadata),
    )
    return _save(figure, args.out)


def _classes(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.results)
    source = metadata["config"]["input"]["source"]
    kind = "noise" if source == "random" else damage(metadata)
    return _save(plot_classes(load_summary(args.results), damage=kind), args.out)


def _stable(args: argparse.Namespace) -> int:
    table = pl.concat([load_stability(results) for results in args.results])
    return _save(plot_stable(table), args.out)


def _fields(args: argparse.Namespace) -> int:
    return _save(plot_fields(load_fields(args.results), args.datasets, log=args.log), args.out)


def _unstable(args: argparse.Namespace) -> int:
    if args.registry is None:
        registry = Registry.discover(Path.cwd())
    else:
        registry = Registry.from_file(args.registry)
    patterns = registry.load(args.dataset)
    fields = aligned_fields(
        load_fields(args.results), args.dataset, args.learning, patterns.shape[1:]
    )
    if fields.shape != patterns.shape:
        raise ValueError(
            f"the results have {fields.shape[0]} patterns of `{args.dataset}`, "
            f"the registry {patterns.shape[0]}"
        )
    mask = unstable_mask(patterns, fields, zero_field(args.results))
    return _save(plot_unstable(patterns, mask, columns=args.columns), args.out)


def _overlaps(args: argparse.Namespace) -> int:
    return _save(plot_overlaps(load_datasets(args.results), args.datasets), args.out)


def _capacity(args: argparse.Namespace) -> int:
    return _save(plot_capacity(load_capacity_summary(args.results)), args.out)


def _outcomes(args: argparse.Namespace) -> int:
    return _save(plot_outcomes(load_summary(args.results)), args.out)


def _attractors(args: argparse.Namespace) -> int:
    shapes = bitmap_shapes(load_metadata(args.results))
    if args.dataset not in shapes:
        raise ValueError(f"no dataset `{args.dataset}` in the experiment")
    figure = plot_attractors(
        load_attractors(args.results), shapes[args.dataset], args.dataset, top=args.top
    )
    return _save(figure, args.out)


def _accessibility(args: argparse.Namespace) -> int:
    figure = plot_accessibility(load_attractors(args.results), args.datasets, mode=args.mode)
    return _save(figure, args.out)


def _distinct(args: argparse.Namespace) -> int:
    figure = plot_distinct(
        load_attractors(args.results),
        load_summary(args.results),
        dataset_sizes(load_metadata(args.results)),
    )
    return _save(figure, args.out)


def _energies(args: argparse.Namespace) -> int:
    figure = plot_energies(load_trials(args.results), args.datasets, mode=args.mode)
    return _save(figure, args.out)
