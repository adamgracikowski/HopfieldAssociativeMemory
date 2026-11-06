# Test fixtures

Artifacts written by the compute layer, so the tests check the real file formats without building `compute`.

- `recall/`: `compute recall` with `recall.toml` (`small-7x7`, Hebbian learning, pattern 2 with 14 flipped pixels, asynchronous updates with a frame every 7 updates; 10 frames). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- recall --config ../analysis/tests/fixtures/recall.toml --out ../analysis/tests/fixtures/recall --force
  ```
- `train/`: `compute train` with `train.toml` (Hebbian weights of `small-7x7`, dense backend, `weights.npy` of 49 x 49). Regenerate from `src/compute`:

  ```bash
  cargo run --release -- train --config ../analysis/tests/fixtures/train.toml --out ../analysis/tests/fixtures/train --force
  ```

The tests check the shapes and statistics of the fixture, not the time or the build in `metadata.json`.
