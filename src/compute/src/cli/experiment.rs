//! The `experiment` subcommand.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use compute::experiment::{ExperimentConfig, ExperimentRun};
use compute::run::OutputDir;

const RESULTS_DIR: &str = "results";

#[derive(Debug, Args)]
pub struct ExperimentArgs {
    /// Experiment configuration (TOML).
    #[arg(long)]
    config: PathBuf,

    /// Output directory (default: `results` next to the configuration).
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,

    /// Overwrite files in a non-empty output directory.
    #[arg(long)]
    force: bool,
}

impl ExperimentArgs {
    /// Runs the experiment, printing training progress to stderr.
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
