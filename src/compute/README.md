# compute

Rust command-line tool that trains Hopfield networks, runs their dynamics and writes every result to files. Plotting is done separately by [`hopfield-analysis`](../analysis/README.md).

## Build

The toolchain is pinned in `rust-toolchain.toml` and installed by `rustup`. Run all commands in `src/compute`:

```bash
cargo build --release
cargo run --release -- --help
cargo run --release -- --version   # crate version, git commit, dirty flag
```

## Subcommands

| Subcommand | What it does | Input | Output |
|---|---|---|---|
| `inspect` | Prints size, pixel fractions, overlaps and bitmaps of a dataset | dataset name | terminal |
| `convert-images` | Converts photos to bipolar bitmaps (median threshold, Floyd-Steinberg dithering) | `data/images/<set>/config.toml` | `.npy` datasets + provenance `.json` |
| `generate` | Creates datasets of groups of similar patterns (prototypes with flipped pixels) | `data/generated/<set>/config.toml` | `.npy` datasets + provenance `.json` |
| `train` | Trains a network and checks which stored patterns are fixed points | run config | `metadata.json`, weights `.npy` |
| `recall` | Runs one recall and records every visited state | run config | `metadata.json`, `history.npy`, `frames.json` |
| `experiment` | Grid of datasets x rules x update modes x damage levels, many trials per cell | experiment config | `trials.parquet`, `summary.json`, `datasets.json`, optional `stability.parquet`, `attractors.parquet` |
| `capacity` | Stability of growing pattern sets (dataset prefixes and random sets), no dynamics | capacity config | `capacity.parquet`, `summary.json` |
| `benchmark` | Time and memory of training and recall versus `N`, dense and factored weights | benchmark config | `benchmark.parquet` |
| `search-stable` | Searches large sets of small bitmaps (up to 32 pixels) that are strictly stable with both rules | search config | `summary.json`, basis and sample datasets |

Every command that writes a directory also writes `metadata.json` (command, time, build, the configuration with all defaults). Output directories must be empty or new, `--force` overwrites. `experiment`, `capacity`, `benchmark` and `search-stable` write to `results/` next to their configuration unless `--out` is given.

```bash
cargo run --release -- inspect letters-14x20 --pattern 0 --pattern 1
cargo run --release -- recall --config ../../data/experiments/cats-occlusion/recall-occlusion-oja.toml --out ../../data/experiments/cats-occlusion/runs/recall-occlusion-oja --force
cargo run --release -- experiment --config ../../data/experiments/noise/config.toml
```

## Configuration

Configurations are TOML files. Relative paths are resolved against the file's directory, and datasets are looked up by name in `data/datasets/datasets.toml`. Every experiment in `data/experiments/` has its configuration there, with the commands that produce its results and figures in the header comments.

A run configuration (`train`, `recall`):

```toml
dataset = "small-7x7"           # registered dataset
patterns = 3                    # optional: only the first 3 patterns
seed = 42                       # required by recall
backend = "auto"                # dense up to N = 5792, then factored

[learning]
rule = "oja"                    # or "hebb"; Oja: eta, tolerance, max_epochs, variant

[dynamics]                      # optional, defaults shown
mode = "asynchronous"           # or "synchronous"
zero_field = "keep"             # sgn(0) keeps the state, or "plus-one"
max_steps = 1000

[input]                         # recall only
source = "pattern"              # damaged stored pattern; also "occlusion", "random", "file"
index = 0
flips = 5
```

An experiment configuration:

```toml
seed = 7
datasets = ["small-7x7", { name = "animals-14x9", patterns = 3 }]
trials = 100                    # trials per cell

[[learning]]
rule = "hebb"
[[learning]]
rule = "oja"

[dynamics]
modes = ["synchronous", "asynchronous"]

[input]
source = "pattern"              # or "occlusion" (with occlusion = [...]) or "random"
noise = [0.0, 0.1, 0.2]         # fractions of flipped pixels
```

The other commands are configured the same way. Their full key lists are in the doc comments of the configuration types, and `data/experiments/capacity`, `benchmark` and `stable-5x5` and `data/generated/` and `data/images/` hold working examples.

## Design

- **Weights.** Dense weights store the `N x N` matrix. Factored weights store `W = A Xi - diag(d)` with an `N x p` matrix `A`, so memory and a field evaluation are `O(Np)`. Both rules produce weights of this form, which makes 60 000-neuron networks practical (6.6 MiB instead of 26.8 GiB).
- **Oja's rule.** The subspace rule is trained in batches on `p x p` matrices, so an epoch costs `O(p^3)` independently of `N`.
- **Determinism.** Every run derives its own `ChaCha8Rng` seed from the configuration seed and its position in the grid. Fields are computed row by row in a fixed order, so results do not depend on the number of threads. Running a command again gives identical files apart from the time and build in `metadata.json`. An experiment trial can be replayed with `recall`.
- **Outcomes.** A final state is classified by exact equality as `pattern`, `negation`, `spurious`, `cycle` (two-cycle) or `unconverged`. Fractions come with 95% Wilson intervals.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

The tests include the hand-checked small networks and compare the committed results in `data/experiments/` byte for byte with a fresh computation.
