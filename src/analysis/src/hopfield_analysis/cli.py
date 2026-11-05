"""Command-line interface ``hopfield-analysis``."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from hopfield_analysis.experiments import load_summary, plot_recovery


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command given by ``argv``.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` if
            ``None``.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(
        prog="hopfield-analysis",
        description="Plots of the results written by the compute layer.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    recovery = commands.add_parser(
        "recovery",
        help="plot the recovered fraction against the noise level of an experiment",
    )
    recovery.add_argument("results", type=Path, help="results directory of an experiment")
    recovery.add_argument("--out", type=Path, required=True, help="output image (PNG)")
    args = parser.parse_args(argv)

    figure = plot_recovery(load_summary(args.results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"written to {args.out}")
    return 0
