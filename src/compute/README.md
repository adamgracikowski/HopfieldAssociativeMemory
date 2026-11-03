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

## Checks

The same checks run in CI:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
