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

## Checks

The same checks run in CI:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
