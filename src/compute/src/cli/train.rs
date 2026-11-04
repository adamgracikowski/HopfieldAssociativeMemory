//! The `train` subcommand.

use std::path::PathBuf;

use clap::Args;
use compute::run::{OutputDir, RunConfig, TrainRun};

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Run configuration (TOML) with the dataset and the learning rule; the
    /// `sgn(0)` convention of its dynamics decides pattern stability.
    #[arg(long)]
    config: PathBuf,

    /// Output directory; it must be empty or not exist yet.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,

    /// Write into a non-empty output directory, replacing files of the same
    /// name.
    #[arg(long)]
    force: bool,
}

impl TrainArgs {
    /// Trains the configured network, writes its weights and metadata and
    /// prints a summary.
    pub fn run(self) -> compute::Result<()> {
        let config = RunConfig::from_file(&self.config)?;
        let out = OutputDir::create(&self.out, self.force)?;
        let run = TrainRun::execute(&config)?;
        run.write(&config, &out)?;

        let dataset = &run.dataset;
        let training = &run.trained;
        println!(
            "dataset      {} ({}x{}, p = {}, N = {})",
            dataset.name, dataset.width, dataset.height, dataset.patterns, dataset.neurons
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
