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

The tests check the shapes and statistics of the fixture, not the time or the build in `metadata.json`. A test of `src/compute` (`tests/cli.rs`) runs both configurations and compares the data files with these fixtures, so a change of the output format fails there until the fixtures are regenerated.
