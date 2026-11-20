//! The `train` subcommand.

use std::path::PathBuf;

use clap::Args;
use compute::run::{OutputDir, RunConfig, TrainRun};

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Run configuration (TOML).
    #[arg(long)]
    config: PathBuf,

    /// Output directory.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,

    /// Overwrite files in a non-empty output directory.
    #[arg(long)]
    force: bool,
}

impl TrainArgs {
    /// Trains the network and prints a summary.
    pub fn run(self) -> compute::Result<()> {
        let config = RunConfig::from_file(&self.config)?;
        let out = OutputDir::new(&self.out, self.force)?;
        let run = TrainRun::execute(&config)?;
        run.write(&config, &out)?;

        let dataset = &run.dataset;
        let training = &run.trained;
        println!(
            "dataset      {} ({}x{}, p = {}, N = {})",
            dataset.label(),
            dataset.width,
            dataset.height,
            dataset.patterns,
            dataset.neurons
        );
        println!(
            "learning     {}, {} backend",
            config.learning,
            training.weights.backend()
        );
        let convergence = if training.converged {
            "converged"
        } else {
            "not converged"
        };
        println!("epochs       {} ({convergence})", training.epochs);
        println!(
            "stable       {} of {} patterns",
            run.stability.stable, dataset.patterns
        );
        println!("written to   {}", out.path().display());
        Ok(())
    }
}
