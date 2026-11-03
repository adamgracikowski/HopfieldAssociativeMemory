# HopfieldAssociativeMemory
Hopfield associative memory network built from scratch for the Artificial Intelligence Methods course. Features Hebbian and Oja's learning rules for pattern recognition. Includes synchronous/asynchronous modes and step-by-step bitmap visualization.

## Repository layout

| Path | Contents |
|---|---|
| `src/compute` | Compute layer (Rust): training, simulation and experiments, results written to files. See [`src/compute/README.md`](src/compute/README.md). |
| `src/analysis` | Presentation layer (Python, uv): visualization of the results. See [`src/analysis/README.md`](src/analysis/README.md). |
| `data/datasets` | Training pattern sets: one pattern per CSV row, pixel values `1` and `-1`. |
| `data/experiments` | Experiment configurations and results. |
| `docs` | Report (LaTeX) and its build scripts. |
