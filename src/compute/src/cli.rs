//! Parsing of the command line and dispatch to the subcommands.

mod convert_images;
mod inspect;
mod recall;
mod train;

use clap::{Parser, Subcommand};
use compute::build_info;

use convert_images::ConvertImagesArgs;
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
            Command::Train(args) => args.run(),
            Command::Recall(args) => args.run(),
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Show the size, statistics and bitmaps of a registered dataset.
    Inspect(InspectArgs),

    /// Convert photographs to bipolar bitmap datasets saved as `.npy`.
    ConvertImages(ConvertImagesArgs),

    /// Train a network on a dataset and save its weights and the stability
    /// of the patterns.
    Train(TrainArgs),

    /// Train a network, run it from an initial state and save every visited
    /// state.
    Recall(RecallArgs),
}
