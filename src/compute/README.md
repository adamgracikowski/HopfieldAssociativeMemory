# compute

Compute layer of the project: a Rust command-line tool that trains Hopfield networks, simulates their dynamics and writes the results to files.

## Requirements

Rust toolchain from `rust-toolchain.toml` (installed automatically by `rustup`).

## Usage

Run all commands in `src/compute`.

```bash
cargo build --release
cargo run --release -- --help
cargo run --release -- --version
```

`--version` prints the crate version, the git commit of the sources and whether the crate files differed from that commit at build time.

## Subcommands

### `inspect`

Prints the size, the fraction of `1` pixels per pattern, the overlap matrix and the bitmaps (`#` for `1`, `.` for `-1`) of a dataset from `data/datasets/datasets.toml`.

```bash
cargo run --release -- inspect small-7x7
cargo run --release -- inspect letters-14x20 --pattern 0 --pattern 1
cargo run --release -- inspect large-25x50 --no-render
```

The registry is searched for in the current directory and its parents. Pass `--registry <path>` to use another one.

### `convert-images`

Converts photographs to bipolar bitmaps and writes them as `.npy` datasets (`int8`, shape `(p, height, width)`), one per binarization method. Each image is converted to grayscale, scaled with the centre cropped to fill `width x height` and binarized: dark pixels become `1`, light pixels `-1`.

```bash
cargo run --release -- convert-images --config ../../data/images/cats/config.toml
cargo run --release -- convert-images --config ../../data/images/cats/config.toml --preview target/previews
```

The configuration (paths relative to its directory):

```toml
width = 300
height = 200
sources = "sources.tsv"   # tab-separated: file, sha1, url, page, author, license, license_url
images = "raw"            # directory with the source images

[[outputs]]
method = "median"         # threshold at the median brightness
file = "../../datasets/cats-300x200-median.npy"

[[outputs]]
method = "floyd-steinberg" # error diffusion dithering
file = "../../datasets/cats-300x200-dither.npy"
```

Patterns follow the order of the source list. Every image must be a plain file name in `images` and match its SHA-1 checksum. Outputs must have distinct file names. Next to every `.npy` file a `.json` file of the same name records its provenance: the build (version, commit, dirty flag), the configuration file relative to the dataset, the bitmap size, the binarization and the source images with their checksums, authors and licenses. It has no timestamp, so converting again with the same code and images gives the same file. `--preview <dir>` also writes every bitmap as a black-and-white PNG to `<dir>/<output file stem>/`.

### `train` and `recall`

Both commands read a run configuration (TOML) and write their results to an output directory, which must be empty or not exist yet (`--force` writes into a non-empty one, replacing files of the same name).

```bash
cargo run --release -- train --config run.toml --out runs/train
cargo run --release -- recall --config run.toml --out runs/recall --render
```

The configuration (relative paths are resolved against its directory):

```toml
dataset = "small-7x7"     # name in the dataset registry
# registry = "../../datasets/datasets.toml"  # by default found from this file's directory upwards
seed = 42                 # required by `recall`
backend = "auto"          # `auto` (dense up to N = 5792), `dense` or `factored`

[learning]
rule = "oja"              # `hebb`, or `oja` with optional parameters:
variant = "subspace"      # `subspace` or `single-neuron`
eta = 0.5
max_epochs = 100000
tolerance = 1e-6

[dynamics]                # optional, defaults shown
mode = "asynchronous"     # or `synchronous`
zero_field = "keep"       # sgn(0): keep the previous state, or `plus-one`
order = "permutation"     # asynchronous order: `permutation`, `random` or `sequential`
max_steps = 1000          # synchronous steps or asynchronous sweeps
# record_every = 49       # asynchronous frame interval in neuron updates, one sweep by default

[input]                   # required by `recall`
source = "pattern"        # stored pattern `index` with `flips` random pixels flipped
index = 0
flips = 5
# source = "random"       # random state
# source = "file"         # pattern `index` of an `.npy` set of the same bitmap size,
# path = "x.npy"          # given with `path` (and `index`, 0 by default)
```

The seed starts separate random streams for the initial state and for the dynamics, so the same configuration gives the same results.

`train` learns the weights and checks which stored patterns are fixed points (with the `zero_field` convention). It writes:

- `metadata.json`: command, time, build (version, commit, dirty flag as `true`, `false` or `null`), the configuration with all defaults and its paths as written, the dataset (with the registry relative to the configuration directory), the training result (backend, epochs, convergence) and the number of unstable neurons of every pattern,
- `weights.npy` (`float64`, `N x N`) for dense weights, or `coefficients.npy` (`float64`, `N x p`) for factored weights `W = A Xi - diag(d)`, where `Xi` holds the dataset patterns as rows and `d_i = sum_mu A[i][mu] Xi[mu][i]`. The other of the two files is removed, so a directory never holds weights of two runs.

`recall` trains the network the same way, runs it from the initial state and writes:

- `metadata.json`: as for `train`, with `outcome` (steps, changes, stop reason, final energy, number of frames, target pattern, its Hamming distance to the initial and the final state, and the nearest pattern with its overlap),
- `history.npy` (`int8`, shape `(frames, height, width)`): the initial state, the states between steps and the final state, consecutive frames always different,
- `frames.json`: `energy`, `changes` (neurons changed since the previous frame) and `overlaps` (with every stored pattern) of each frame.

Running `recall` again with the same configuration and code gives identical files, apart from `created` in `metadata.json`.

### `experiment`

Runs a grid of datasets, learning rules, update modes and noise levels with many trials in every cell, in parallel over the trials.

```bash
cargo run --release -- experiment --config ../../data/experiments/00-example/config.toml
```

The results go to `results/` next to the configuration, or to `--out <dir>`; the directory must be empty or not exist yet (`--force` as above). The configuration:

```toml
seed = 7                  # master seed, required
datasets = ["small-7x7", "animals-14x9"]  # grid axis
# registry = "../../datasets/datasets.toml"  # by default found from this file's directory upwards
backend = "auto"          # as in a run configuration
trials = 100              # trials per cell

[[learning]]              # grid axis, one table per rule, parameters as above
rule = "hebb"
[[learning]]
rule = "oja"

[dynamics]                # optional, defaults shown
modes = ["asynchronous"]  # grid axis
zero_field = "keep"
order = "permutation"
max_steps = 1000

[input]
source = "pattern"        # in every trial each stored pattern with flipped pixels
noise = [0.0, 0.1, 0.2]   # grid axis: fractions of flipped pixels, round(noise * N) pixels
# source = "random"       # in every trial one random state
```

Axes must not be empty or repeat a value, and the rule names (`hebb`, `oja (subspace)`, `oja (single-neuron)`) must differ. Every (dataset, rule) pair is trained once; its stability check uses `zero_field`.

Every run draws from its own seed: the first 8 bytes of the SHA-1 of `seed/dataset/source/flips/target/trial` (`-` for the flips and target of a random input). It does not depend on the rule or the mode, so all of them start from the same initial states, and the results do not depend on the number of threads. A `recall` run with this seed, the same dataset, rule and mode, and `[input]` `source = "pattern"`, `index = target`, `flips = flips` repeats the run with its history.

The final state is classified by exact equality: `pattern` (a fixed point equal to a stored pattern), `negation` (equal to a negated pattern), `spurious` (another fixed point), `cycle` (two-cycle) or `unconverged` (step limit). The command writes:

- `trials.parquet`: one row per run, ordered by cell, trial and target, with the columns `dataset`, `learning`, `mode`, `noise`, `flips`, `target`, `trial`, `seed`, `steps`, `changes`, `termination`, `class`, `energy`, `recovered` (final state equals the target), `initial_distance`, `final_distance` (Hamming distances to the target), `final_overlap` (with the target), `nearest` and `nearest_overlap` (the pattern with the largest `|m|`); target columns are null for random inputs,
- `summary.json`: `{"cells": [...]}` with, for every cell, the number of runs, the fraction of every class and of recovered runs with a 95% Wilson interval (`count`, `fraction`, `lower`, `upper`), and the mean and sample standard deviation of steps, final energy and final overlap,
- `metadata.json`: command, time, build, the configuration with all defaults, the datasets and every training (backend, epochs, convergence, unstable neurons of every pattern).

Running the experiment again gives identical `trials.parquet` and `summary.json`; `metadata.json` differs only in `created`.

## Checks

The same checks run in CI:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
