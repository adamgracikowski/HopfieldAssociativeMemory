//! Command-line interface of the compute layer.

mod cli;

use clap::Parser;
use peak_alloc::PeakAlloc;

/// Global allocator that counts the bytes on the heap and their peak, read
/// by the `benchmark` subcommand; it allocates through the system
/// allocator.
#[global_allocator]
static HEAP: PeakAlloc = PeakAlloc;

fn main() -> miette::Result<()> {
    cli::Cli::parse().run()?;
    Ok(())
}
