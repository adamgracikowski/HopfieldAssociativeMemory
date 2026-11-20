//! Command-line interface of the compute layer.

mod cli;

use clap::Parser;
use peak_alloc::PeakAlloc;

/// Global allocator tracking the heap peak for `benchmark`.
#[global_allocator]
static HEAP: PeakAlloc = PeakAlloc;

fn main() -> miette::Result<()> {
    cli::Cli::parse().run()?;
    Ok(())
}
