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

### `generate`

Generates datasets of groups of similar patterns: every group holds variants of one prototype, a pattern of a registered dataset, with a fraction of the pixels flipped at random.

```bash
cargo run --release -- generate --config ../../data/generated/groups-25x25/config.toml
```

The configuration (paths relative to its directory):

```toml
seed = 20261110           # master seed, required
dataset = "large-25x25"   # registered dataset with the prototypes
# registry = "../../datasets/datasets.toml"  # by default found from this file's directory upwards
prototypes = [0, 2, 4]    # indices of the prototypes, one group each
variants = 5              # patterns per group

[[outputs]]
flip = 0.05               # fraction of flipped pixels, round(flip * N) pixels per variant
file = "../../datasets/groups-25x25-d05.npy"
```

Every output holds `prototypes.len() * variants` patterns, group by group; the prototypes themselves are not stored. A variant with `k` flipped pixels has overlap exactly `1 - 2k/N` with its prototype, two variants of a group about `(1 - 2k/N)^2`. Variant `v` of prototype `mu` draws from the seed of `seed/<output file stem>/mu/v`. Prototypes must not repeat, `flip` must lie in `[0, 1]` and outputs must have distinct file names. Next to every `.npy` file a `.json` file records its provenance (build, configuration relative to the dataset, seed, source dataset, prototypes, variants, `flip` and flipped pixels, bitmap size), without a timestamp, so generating again gives the same files. The datasets are added to the registry by hand.

### `train` and `recall`

Both commands read a run configuration (TOML) and write their results to an output directory, which must be empty or not exist yet (`--force` writes into a non-empty one, replacing files of the same name).

```bash
cargo run --release -- train --config run.toml --out runs/train
cargo run --release -- recall --config run.toml --out runs/recall --render
```

The configuration (relative paths are resolved against its directory):

```toml
dataset = "small-7x7"     # name in the dataset registry
patterns = 3              # optional: only the first 3 patterns
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
# source = "occlusion"    # stored pattern `index` with a band of `rows` consecutive
# rows = 3                # full-width rows at a random position set to -1
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

Runs a grid of datasets, learning rules, update modes and damage levels (noise or occlusion) with many trials in every cell, in parallel over the trials.

```bash
cargo run --release -- experiment --config ../../data/experiments/00-example/config.toml
```

The results go to `results/` next to the configuration, or to `--out <dir>`; the directory must be empty or not exist yet (`--force` as above). The configuration:

```toml
seed = 7                  # master seed, required
datasets = ["small-7x7", { name = "animals-14x9", patterns = 3 }]  # grid axis; a table
                                          # stores only the first patterns
# registry = "../../datasets/datasets.toml"  # by default found from this file's directory upwards
backend = "auto"          # as in a run configuration
trials = 100              # trials per cell
stability_fields = false  # optional: write stability.parquet
attractors = false        # optional: write attractors.parquet

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
# source = "occlusion"    # in every trial each stored pattern with a band of rows set to -1
# occlusion = [0.0, 0.3]  # grid axis: fractions of covered rows, round(occlusion * H) rows
# source = "random"       # in every trial one random state
```

Axes must not be empty or repeat a value, and the rule names (`hebb`, `oja (subspace)`, `oja (single-neuron)`) must differ. Two noise levels must not give the same number of flipped pixels on any dataset (e.g. `0.1` and `0.11` on `N = 49` both flip 5), and two occlusion levels not the same number of covered rows (e.g. `0.1` and `0.2` of 7 rows both cover 1), because they would give identical cells. A prefix (`{ name, patterns }`, from 1 to the size of the set) appears in the results as `name:p`, e.g. `animals-14x9:3`, and its seeds use the registry name, so a target gets the same damaged input in every prefix of a dataset. Every (dataset, rule) pair is trained once; its stability check uses `zero_field`. A synchronous cell without damaged pixels does not depend on random numbers, so it runs a single trial and its fractions are exact (`deterministic` in the summary).

Every run draws from its own seed: the first 8 bytes of the SHA-1 of `seed/dataset/source/pixels/target/trial`, where `pixels` is the number of damaged pixels (`-` for the pixels and target of a random input). It does not depend on the rule or the mode, so all of them start from the same initial states, and the results do not depend on the number of threads. A `recall` run repeats a run with its history when it has this seed, the experiment's `dataset` (with `patterns = p` for a prefix), `registry` and `backend`, the `[learning]` table of the rule, `[dynamics]` with the cell's `mode` and the experiment's `zero_field`, `order` and `max_steps`, and `[input]` `source = "pattern"`, `index = target`, `flips = flips` (`source = "occlusion"`, `index = target`, `rows = flips / width` for occlusion, or `source = "random"`).

The final state is classified by exact equality: `pattern` (a fixed point equal to a stored pattern), `negation` (equal to a negated pattern), `spurious` (another fixed point), `cycle` (two-cycle) or `unconverged` (step limit). The command writes:

- `trials.parquet`: one row per run, ordered by cell, trial and target, with the columns `dataset`, `learning`, `mode`, `noise` (the damage level: fraction of flipped pixels or of covered rows), `flips` (damaged pixels: flipped, or covered, rows times width; only covered pixels equal to `1` change), `target`, `trial`, `seed`, `steps`, `changes`, `termination`, `class`, `energy`, `recovered` (the run ended in a fixed point equal to the target), `initial_distance`, `final_distance` (Hamming distances to the target), `final_overlap` (with the target), `nearest` and `nearest_overlap` (the pattern with the largest `|m|`); target columns are null for random inputs,
- `summary.json`: `{"cells": [...]}` with, for every cell, the number of runs, whether it is `deterministic`, the fraction of every class, of recovered runs and of runs whose final state is nearest their target (`nearest_target`: the target has `m > 0` and the largest `|m|` of all stored patterns, ties included, in any final class, so it also counts runs that end near an unstable target) with a 95% Wilson interval (`count`, `fraction`, `lower`, `upper`; both ends equal the fraction in a deterministic cell), and the mean and sample standard deviation of steps, final energy, final overlap and final Hamming distance to the target (`final_distance`),
- `datasets.json`: `{"datasets": [...]}` with, for every dataset, its size, the load `p/N`, the fraction of `1` in every pattern (`fraction_on`), the overlap matrix `m^{mu nu}` (`overlaps`), for every pattern the sum of `|m^{mu nu}|` over the other patterns (`overlap_sums`; with Hebbian weights a pattern whose sum is below `1 - p/N` is stable) and the largest `|m^{mu nu}|` of two different patterns with the pair,
- `stability.parquet` (only with `stability_fields = true`): the aligned local field `xi_i^mu h_i` (`h = W xi^mu`) of every neuron of every stored pattern, one row per training, pattern and neuron, with the columns `dataset`, `learning`, `pattern`, `neuron` (row-major pixel index) and `field`; a negative field means the neuron would change, a zero one is decided by `zero_field`. It has `p N` rows per rule, so it is off by default,
- `attractors.parquet` (only with `attractors = true`): the distinct final states of every cell, one row per cell and state, ordered by cell and then by `count` (ties by the first run), with the columns `dataset`, `learning`, `mode`, `noise`, `rank`, `count` (runs that ended in the state), `fraction` (of the runs of the cell), `class`, `termination`, `energy` (computed from the weights), `nearest`, `nearest_overlap`, `overlaps` (list, overlap with every stored pattern), the closest mixture of stored patterns, `target`, `trial` and `seed` of the first run that ended in the state (for `recall`) and `state` (list of `1`/`-1`, row-major). A cycle of length 2 is one row, represented by the lexicographically smaller of its two states. The mixture of size `k` (3, 5 or 7, at most `p`) is `sgn(sum e_mu xi^mu)` over the `k` patterns with the largest `|m|`, with the signs `e_mu` of their overlaps: `mixture_size`, `mixture_distance` (Hamming distance to the state), `mixture_patterns` and `mixture_signs` describe the closest one (the smaller on ties), and `mixture` is its size when the state equals it (null otherwise, and all mixture columns are null for `p < 3`). For strongly correlated patterns the check may miss a mixture of patterns with smaller overlaps, so the distance is an upper bound,
- `metadata.json`: command, time, build, the configuration with all defaults, the datasets and every training (backend, epochs, convergence, unstable neurons of every pattern).

Running the experiment again gives identical `trials.parquet`, `summary.json`, `datasets.json`, `stability.parquet` and `attractors.parquet`; `metadata.json` differs only in `created`.

### `capacity`

Counts the stable stored patterns as their number `p` grows, without running the dynamics: for the prefixes of registered datasets (the first 1, 2, ..., `p` patterns) and for sets of independent random patterns, in parallel over the sets.

```bash
cargo run --release -- capacity --config ../../data/experiments/08-capacity/config.toml
```

The results go to `results/` next to the configuration, or to `--out <dir>` (`--force` as above). The configuration:

```toml
seed = 7                  # master seed, required
datasets = ["letters-14x20"]  # optional: prefixes of these datasets
# registry = "../../datasets/datasets.toml"  # by default found from this file's directory upwards
backend = "auto"          # as in a run configuration
zero_field = "keep"       # sgn(0) in the stability check
trials = 50               # random sets for every (N, p)

[[random]]                # optional, one table per number of neurons
neurons = 280
patterns = { start = 2, end = 80, step = 2 }  # p = 2, 4, ..., 80; step 1 by default

[[learning]]              # one table per rule, parameters as above
rule = "hebb"
[[learning]]
rule = "oja"
```

There must be at least one dataset or random entry; datasets, numbers of neurons and rule names must not repeat, and a random set has at most `N` patterns. A dataset prefix is fixed, so it has a single trial. The patterns of a random set are drawn from the input stream of the seed derived from `seed/random/N/p/trial` (as in `experiment`); it does not depend on the rule, so every rule sees the same sets. The command writes:

- `capacity.parquet`: one row per pattern set and rule, ordered by source (datasets, then random entries), `p`, trial and rule, with the columns `source` (dataset name or `random`), `learning`, `neurons`, `patterns`, `trial`, `seed` (null for a prefix), `stable`, `unstable_neurons` (over all patterns), `epochs` and `converged`,
- `summary.json`: `{"rows": [...]}` with, for every source, `p` and rule, the load `p/N`, the number of trials, whether it is `deterministic`, the mean and sample standard deviation of the fraction of stable patterns (`stable_fraction`) and of neurons that would change (`bit_errors`), the fraction of sets with all patterns stable with a 95% Wilson interval (`all_stable`, exact for a prefix) and, for random sets with the Hebbian rule, the signal-to-noise prediction (`theory`: `bit_error = Phi(-sqrt(N/p))` and `pattern_stable = (1 - bit_error)^N`; the large-`N` form of MacKay 2003, eqs. 42.18-42.22, which assumes independent neurons and ignores the zero diagonal, so it slightly overestimates the errors of the measured sets),
- `metadata.json`: command, time, build, the configuration with all defaults and the datasets.

Running it again gives identical `capacity.parquet` and `summary.json`; `metadata.json` differs only in `created`. Oja's subspace rule on random sets is the slow part: its epochs cost `O(p^3)` and their number grows with `p`.

### `benchmark`

Measures the time and memory of training and recall for registered datasets and for random sets of growing `N`, with dense and factored weights, one measurement after another (only the local fields of a large network are computed in parallel).

```bash
cargo run --release -- benchmark --config ../../data/experiments/13-benchmark/config.toml
```

The results go to `results/` next to the configuration, or to `--out <dir>` (`--force` as above). The configuration:

```toml
seed = 7                  # master seed, required
datasets = ["cats-300x200-median"]  # optional: registered datasets
# registry = "../../datasets/datasets.toml"  # by default found from this file's directory upwards
repeats = 5               # measurements of every (set, backend, rule)
backends = ["dense", "factored"]  # `auto` is not allowed
dense_limit = 16384       # largest N measured with dense weights (8 N^2 bytes)
noise = 0.1               # fraction of flipped pixels of the recall input

[random]                  # optional: one random set of `patterns` patterns per N
neurons = [1024, 4096, 16384, 60000]
patterns = 12

[[learning]]              # one table per rule, parameters as above
rule = "hebb"
[[learning]]
rule = "oja"
```

There must be at least one dataset or number of neurons, without repeats, and at least one set measured with some backend; a random set has at most `N` patterns and is drawn as in `capacity` (trial 0). Every measurement trains the network, computes the local fields of one random state (the main cost of a synchronous step; timed after one untimed call, so it does not include waking up the threads), makes one asynchronous sweep from that state (`N` updates in a random order on a field tracker built before the clock starts, so without the energies and the setup of a recall run), and runs the synchronous and the asynchronous dynamics (default settings) from the first pattern with `round(noise N)` pixels flipped. The state, the input and the update orders come from the seed derived from `seed/source/N/repeat`, so every backend and rule gets the same ones. The binary counts the bytes on the heap with a global allocator (`peak_alloc`); the peak is the largest growth of the heap during training and during both recall runs. The command prints one line per measurement and writes:

- `benchmark.parquet`: one row per measurement, ordered by set (datasets, then random sets), backend, rule and repeat, with the columns `source`, `neurons`, `patterns`, `backend`, `learning`, `repeat`, `epochs`, `train_seconds`, `weights_bytes` (`8 N^2` dense, `p N + 8 N p + 8 N` factored), `train_peak_bytes`, `fields_seconds`, `sweep_seconds`, `sync_seconds`, `sync_steps`, `async_seconds`, `async_sweeps` and `recall_peak_bytes` (times in wall-clock seconds),
- `metadata.json`: command, time, build, the configuration with all defaults, the datasets, the number of threads and whether the heap was measured.

The times depend on the machine and change from run to run; the steps, epochs and bytes of the weights do not.

### `search-stable`

Searches large sets of bitmaps whose every pattern is a strict fixed point (every aligned field `xi_i h_i > 1e-10`, whatever `sgn(0)` means) of the network trained with each rule. The sets are the bipolar points of hyperplanes, `S_u = {xi in {-1, 1}^N : u . xi = 0}` for a vector `u` of positive integers: Oja's subspace rule tends to the projection `P` onto the span of its training set, here `u^perp`, and with the zeroed diagonal every member has the aligned field `1 - P_ii > 0`.

```bash
cargo run --release -- search-stable --config ../../data/experiments/11-stable/config.toml
```

The results go to `results/` next to the configuration, or to `--out <dir>` (`--force` as above). The configuration:

```toml
width = 5                 # bitmap of N = width * height pixels, N from 2 to 32
height = 5
max_weight = 3            # candidates u: non-decreasing, weights 1 to max_weight
verify = 5                # largest non-empty sets checked for stability
samples = 24              # optional, members of the best set written to `sample`
basis = "../../datasets/stable-5x5-basis.npy"    # optional outputs, relative to this file
sample = "../../datasets/stable-5x5-sample.npy"

[oja]                     # optional, parameters of Oja's rule as above
```

Permuting `u` permutes the pixels and changing the sign of `u_j` flips pixel `j`, so the non-decreasing positive vectors cover all sets up to such a change. The command counts `|S_u|` of every candidate by dynamic programming over partial sums, then checks the `verify` largest sets by enumerating all `2^N` states in fixed chunks in parallel (the result does not depend on the number of threads): the Hebbian weights of the whole set, accumulated exactly from the integer sums `sum xi_i xi_j`, and Oja's rule trained (dense weights) on a basis of `N - 1` members, chosen by Gram-Schmidt with pivoting among `8 N` evenly spread members so that it is close to orthogonal. The best set is the largest one strictly stable with both rules. Nothing is random. The command writes:

- `summary.json`: `neurons`, `best` (the best checked set or `null`), `verified` (for every checked set its `weights`, `size`, and for `hebb` and `oja` the number of `patterns`, of strictly `stable` ones and the smallest and mean aligned field; the number of patterns, epochs and convergence of Oja's training) and `candidates` (`weights` and `size` of every candidate, largest first),
- `metadata.json`: command, time, build and the configuration with all defaults,
- the basis and the sample of the best set as `.npy` datasets of `width x height` bitmaps, with a provenance `<stem>.json` next to each (build, configuration, `u`, size of the set, `basis` or `sample`).

Running it again gives identical `summary.json` and datasets; `metadata.json` differs only in `created`.

## Checks

The same checks run in CI:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
