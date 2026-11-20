//! The `capacity` subcommand.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use compute::capacity::{CapacityConfig, CapacityRun};
use compute::run::OutputDir;

const RESULTS_DIR: &str = "results";

#[derive(Debug, Args)]
pub struct CapacityArgs {
    /// Capacity configuration (TOML).
    #[arg(long)]
    config: PathBuf,

    /// Output directory (default: `results` next to the configuration).
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,

    /// Overwrite files in a non-empty output directory.
    #[arg(long)]
    force: bool,
}

impl CapacityArgs {
    /// Runs the capacity check and prints a summary.
    pub fn run(self) -> compute::Result<()> {
        let start = Instant::now();
        let config = CapacityConfig::from_file(&self.config)?;
        let path = self
            .out
            .unwrap_or_else(|| config.directory().join(RESULTS_DIR));
        let out = OutputDir::new(&path, self.force)?;
        let run = CapacityRun::execute(&config)?;
        run.write(&config, &out)?;

        println!(
            "sets         {} pattern sets x {} rules",
            run.rows.len() / run.learning.len(),
            run.learning.len()
        );
        println!("time         {:.2} s", start.elapsed().as_secs_f64());
        println!("written to   {}", out.path().display());
        Ok(())
    }
}
