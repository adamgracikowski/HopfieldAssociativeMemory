# Test fixtures

Artifacts written by the compute layer, so the tests check the real file formats without building `compute`.

- `recall/`: `compute recall` with `recall.toml` (`animals-14x9`, 14 pixels wide and 9 high so a swap of the axes is noticed, Hebbian learning, pattern 4 with 30 flipped pixels, asynchronous updates with a frame every 14 updates; 10 frames). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- recall --config ../analysis/tests/fixtures/recall.toml --out ../analysis/tests/fixtures/recall --force
  ```
- `train/`: `compute train` with `train.toml` (Hebbian weights of `small-7x7`, dense backend, `weights.npy` of 49 x 49). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- train --config ../analysis/tests/fixtures/train.toml --out ../analysis/tests/fixtures/train --force
  ```
- `experiment/`: `compute experiment` with `experiment.toml` (stored patterns of `small-7x7` and `animals-14x9` as inputs, both learning rules, synchronous updates, 2 trials, with `stability.parquet`). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- experiment --config ../analysis/tests/fixtures/experiment.toml --out ../analysis/tests/fixtures/experiment --force
  ```
- `occlusion/`: `compute experiment` with `occlusion.toml` (stored patterns of the first 3 patterns of `small-7x7` and of the whole set with 0 or 2 of the 7 rows covered, both learning rules and update modes, 2 trials). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- experiment --config ../analysis/tests/fixtures/occlusion.toml --out ../analysis/tests/fixtures/occlusion --force
  ```
- `random/`: `compute experiment` with `random.toml` (50 random initial states for the first 3 patterns of `small-7x7` and for the whole set, both learning rules and update modes, with `attractors.parquet`). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- experiment --config ../analysis/tests/fixtures/random.toml --out ../analysis/tests/fixtures/random --force
  ```
- `capacity/`: `compute capacity` with `capacity.toml` (prefixes of `small-7x7` and 5 random sets of 49 neurons for every `p` from 2 to 20 in steps of 2, both learning rules). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- capacity --config ../analysis/tests/fixtures/capacity.toml --out ../analysis/tests/fixtures/capacity --force
  ```

- `benchmark/`: `compute benchmark` with `benchmark.toml` (`small-7x7` and random sets of 3 patterns for `N` = 64 and 128, dense weights up to `N` = 64 and factored weights, both learning rules, 2 repeats). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- benchmark --config ../analysis/tests/fixtures/benchmark.toml --out ../analysis/tests/fixtures/benchmark --force
  ```

The tests check the shapes and statistics of the fixtures, not the time or the build in `metadata.json`. A test of `src/compute` (`tests/cli.rs`) runs every configuration except `benchmark.toml` and compares the data files with these fixtures, so a change of the output format fails there until the fixtures are regenerated; the measured times of `benchmark/` change from run to run, so its tests check only the structure.
