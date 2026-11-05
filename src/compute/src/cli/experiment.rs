//! The `experiment` subcommand.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use compute::experiment::{ExperimentConfig, ExperimentRun};
use compute::run::OutputDir;

/// Directory of the results next to the configuration file.
const RESULTS_DIR: &str = "results";

#[derive(Debug, Args)]
pub struct ExperimentArgs {
    /// Experiment configuration (TOML) with the seed, the grid axes and the
    /// number of trials.
    #[arg(long)]
    config: PathBuf,

    /// Output directory, `results` next to the configuration by default; it
    /// must be empty or not exist yet.
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,

    /// Write into a non-empty output directory, replacing files of the same
    /// name.
    #[arg(long)]
    force: bool,
}

impl ExperimentArgs {
    /// Runs every trial of the configured grid, writes the tables and
    /// prints the progress of training to stderr and a summary to stdout.
    pub fn run(self) -> compute::Result<()> {
        let start = Instant::now();
        let config = ExperimentConfig::from_file(&self.config)?;
        let path = self
            .out
            .unwrap_or_else(|| config.directory().join(RESULTS_DIR));
        let out = OutputDir::new(&path, self.force)?;
        let run = ExperimentRun::execute(&config, |training| {
            let stability = &training.stability;
            eprintln!(
                "trained      {}, {}: {} epochs, {} of {} patterns stable",
                training.dataset,
                training.learning,
                training.info.epochs,
                stability.stable,
                stability.unstable_neurons.len()
            );
        })?;
        run.write(&config, &out)?;

        println!(
            "grid         {} cells, {} runs",
            run.cells.len(),
            run.results.len()
        );
        println!("time         {:.2} s", start.elapsed().as_secs_f64());
        println!("written to   {}", out.path().display());
        Ok(())
    }
}
