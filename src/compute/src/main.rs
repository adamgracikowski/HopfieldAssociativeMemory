//! Command-line interface of the compute layer.

use clap::Parser;
use compute::build_info;

/// Hopfield associative memory: training, recall and experiments.
#[derive(Debug, Parser)]
#[command(name = "compute", version = build_info::LONG_VERSION, arg_required_else_help = true)]
struct Cli {}

fn main() -> miette::Result<()> {
    let _cli = Cli::parse();
    Ok(())
}
