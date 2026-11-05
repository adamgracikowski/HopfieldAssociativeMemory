# hopfield-analysis

Presentation layer of the project: a Python package that reads the files written by the compute layer (`src/compute`) and visualizes them.

## Requirements

[uv](https://docs.astral.sh/uv/). It installs Python 3.13 (`.python-version`) and the locked dependencies (`uv.lock`).

## Setup

Run all commands in `src/analysis`.

```bash
uv sync
```

## Usage

```bash
uv run hopfield-analysis recovery ../../data/experiments/00-example/results --out recovery.png
```

`recovery` plots the fraction of recovered runs against the noise level of an experiment written by `compute experiment`: one panel per dataset, one line per learning rule and update mode, with the 95% Wilson intervals from `summary.json`.

The module `hopfield_analysis.experiments` reads the results of an experiment: `load_trials` (`trials.parquet` as a polars DataFrame, one row per run), `load_summary` (the cells of `summary.json` as a flat DataFrame, nested fields joined with `_`, e.g. `recovered_fraction`) and `load_metadata`.

## Checks

The same checks run in CI:

```bash
uv run ruff check
uv run ruff format --check
uv run pytest
```
