# hopfield-analysis

Python package that reads the files written by [`compute`](../compute/README.md) and turns them into figures, GIF animations and an interactive step viewer. It never computes results itself.

## Setup

[uv](https://docs.astral.sh/uv/) installs Python 3.13 (`.python-version`) and the locked dependencies (`uv.lock`). Run all commands in `src/analysis`:

```bash
uv sync
uv run hopfield-analysis --help
```

## Commands

Every figure command takes `--out <path>` without an extension and writes `<path>.pdf` and `<path>.png` (300 dpi).

| Input | Command | Figure |
|---|---|---|
| `compute recall` run | `frames` | Grid of frames with energies (`--max`, `--frames`, `--width`) |
| | `run` | Energy and overlaps with every pattern against the frame |
| | `animate` | Looping GIF of the bitmaps (`--scale`, `--duration`, `--hold`) |
| | `view` | Interactive viewer: slider, arrow keys, `Home`/`End`; `--out` saves a frame instead |
| dataset | `patterns` | Patterns of a registered dataset (`--columns`, `--width`, `--numbers`) |
| `compute train` run | `weights` | Weight matrix as a heat map (dense weights only) |
| `compute experiment` results | `recovery` | Recovered fraction against the noise or occlusion level, with Wilson intervals |
| | `load` | Recovered fraction against the number of patterns of dataset prefixes |
| | `classes` | Final state classes (pattern, negation, spurious, cycle) per damage level |
| | `stable` | Fraction of stable stored patterns per dataset and rule |
| | `fields` | Histograms of the aligned fields (needs `stability_fields = true`) |
| | `unstable` | Patterns with their unstable pixels in red |
| | `overlaps` | Overlap matrices of datasets as heat maps |
| | `outcomes` | Final state classes of random inputs |
| | `attractors` | Most frequent final states of a dataset (needs `attractors = true`) |
| | `accessibility` | How often each stored pattern and its negation is reached |
| | `distinct` | Number of distinct spurious states against the number of patterns |
| | `energies` | Histograms of final energies by class |
| `compute capacity` results | `capacity` | Stable fraction against the load `p/N`, with the theory for random patterns |
| `compute benchmark` results | `benchmark` | Time and memory against `N`, log-log |

```bash
uv run hopfield-analysis view ../../data/experiments/cats-occlusion/runs/recall-occlusion-hebb
uv run hopfield-analysis animate ../../data/experiments/cats-occlusion/runs/recall-occlusion-oja --scale 2 --out ../../data/experiments/cats-occlusion/plots/cat.gif
uv run hopfield-analysis recovery ../../data/experiments/noise/results --out ../../data/experiments/noise/plots/recovery
```

The exact command of every figure in the report is in the comments of its experiment configuration in `data/experiments/`.

## Figure style

Figures match the LaTeX report: Latin Modern Roman at 9 pt (bundled in `src/hopfield_analysis/fonts/` under the GUST Font License), Polish labels and a decimal comma. `--width` gives the printed width as a fraction of the report text width, so the text stays at 9 pt. Bitmaps show `1` as black and `-1` as white.

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run pytest
```

The tests use real `compute` outputs from `tests/fixtures/`.
