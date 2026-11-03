# hopfield-analysis

Presentation layer of the project: a Python package that reads the files written by the compute layer (`src/compute`) and visualizes them.

## Requirements

[uv](https://docs.astral.sh/uv/). It installs Python 3.13 (`.python-version`) and the locked dependencies (`uv.lock`).

## Setup

Run all commands in `src/analysis`.

```bash
uv sync
```

## Checks

The same checks run in CI:

```bash
uv run ruff check
uv run ruff format --check
uv run pytest
```
