//! Command-line interface of the compute layer.

mod cli;

use clap::Parser;

fn main() -> miette::Result<()> {
    cli::Cli::parse().run()?;
    Ok(())
}
