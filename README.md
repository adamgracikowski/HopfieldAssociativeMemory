# HopfieldAssociativeMemory
Hopfield associative memory network built from scratch for the Artificial Intelligence Methods course. Features Hebbian and Oja's learning rules for pattern recognition. Includes synchronous/asynchronous modes and step-by-step bitmap visualization.

## Report

The project report (in Polish) with the theory, the implementation, the experiments and their results: [`docs/report.pdf`](docs/report.pdf).

## Repository layout

| Path | Contents |
|---|---|
| `src/compute` | Compute layer (Rust): training, simulation and experiments, results written to files. See [`src/compute/README.md`](src/compute/README.md). |
| `src/analysis` | Presentation layer (Python, uv): visualization of the results. See [`src/analysis/README.md`](src/analysis/README.md). |
| `data/datasets` | Training pattern sets: one pattern per CSV row (or `.npy` arrays), pixel values `1` and `-1`, listed in `datasets.toml`. |
| `data/images` | Source photos of the large bitmap datasets: download script, source lists with licenses, conversion settings. See [`data/images/README.md`](data/images/README.md). |
| `data/experiments` | Experiment configurations and results. |
| `docs` | Report (LaTeX) and its build scripts. |

## License

The code is licensed under the MIT License (see [`LICENSE`](LICENSE)). The photo-based datasets `data/datasets/cats-*.npy` are not covered by it: each bitmap is an adaptation of a Wikimedia Commons photo and keeps that photo's Creative Commons license, listed in [`data/images/cats/sources.tsv`](data/images/cats/sources.tsv).
