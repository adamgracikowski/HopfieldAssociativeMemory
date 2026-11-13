//! Parsing of the command line and dispatch to the subcommands.

mod benchmark;
mod capacity;
mod convert_images;
mod experiment;
mod generate;
mod inspect;
mod recall;
mod train;

use clap::{Parser, Subcommand};
use compute::build_info;

use benchmark::BenchmarkArgs;
use capacity::CapacityArgs;
use convert_images::ConvertImagesArgs;
use experiment::ExperimentArgs;
use generate::GenerateArgs;
use inspect::InspectArgs;
use recall::RecallArgs;
use train::TrainArgs;

/// Hopfield associative memory: training, recall and experiments.
#[derive(Debug, Parser)]
#[command(name = "compute", version = build_info::LONG_VERSION, arg_required_else_help = true)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

impl Cli {
    /// Runs the chosen subcommand.
    pub fn run(self) -> compute::Result<()> {
        match self.command {
            Command::Inspect(args) => args.run(),
            Command::ConvertImages(args) => args.run(),
            Command::Generate(args) => args.run(),
            Command::Train(args) => args.run(),
            Command::Recall(args) => args.run(),
            Command::Experiment(args) => args.run(),
            Command::Capacity(args) => args.run(),
            Command::Benchmark(args) => args.run(),
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Show the size, statistics and bitmaps of a registered dataset.
    Inspect(InspectArgs),

    /// Convert photographs to bipolar bitmap datasets saved as `.npy`.
    ConvertImages(ConvertImagesArgs),

    /// Generate datasets of groups of similar patterns: variants of stored
    /// pictures with flipped pixels, saved as `.npy`.
    Generate(GenerateArgs),

    /// Train a network on a dataset and save its weights and the stability
    /// of the patterns.
    Train(TrainArgs),

    /// Train a network, run it from an initial state and save every visited
    /// state.
    Recall(RecallArgs),

    /// Run a grid of datasets, learning rules, update modes and damage levels
    /// with many trials and save the trial table and the cell summaries.
    Experiment(ExperimentArgs),

    /// Check how many stored patterns stay stable as their number grows, for
    /// prefixes of datasets and for random patterns.
    Capacity(CapacityArgs),

    /// Measure the time and memory of training and recall for growing
    /// numbers of neurons, with dense and factored weights.
    Benchmark(BenchmarkArgs),
}
