//! The `capacity` subcommand.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use compute::capacity::{CapacityConfig, CapacityRun};
use compute::run::OutputDir;

/// Directory of the results next to the configuration file.
const RESULTS_DIR: &str = "results";

#[derive(Debug, Args)]
pub struct CapacityArgs {
    /// Capacity configuration (TOML) with the seed, the datasets, the random
    /// sets and the learning rules.
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

impl CapacityArgs {
    /// Checks the stability of every configured pattern set with every rule,
    /// writes the tables and prints a summary to stdout.
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
