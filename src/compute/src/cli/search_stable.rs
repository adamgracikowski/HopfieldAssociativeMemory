//! The `search-stable` subcommand.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use compute::run::OutputDir;
use compute::stable::{StableConfig, StableSearch};

/// Directory of the results next to the configuration file.
const RESULTS_DIR: &str = "results";

#[derive(Debug, Args)]
pub struct SearchStableArgs {
    /// Search configuration (TOML) with the bitmap size, the largest weight,
    /// the number of sets to check and the output datasets.
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

impl SearchStableArgs {
    /// Runs the search, writes the summary and the datasets and prints the
    /// checked sets to stdout.
    pub fn run(self) -> compute::Result<()> {
        let start = Instant::now();
        let config = StableConfig::from_file(&self.config)?;
        let path = self
            .out
            .unwrap_or_else(|| config.directory().join(RESULTS_DIR));
        let out = OutputDir::new(&path, self.force)?;
        let search = StableSearch::execute(&config)?;
        search.write(&config, &out)?;

        println!("candidates   {}", search.candidates.len());
        for set in &search.verified {
            println!(
                "u = {:?}  |S_u| = {}  strictly stable: Hebb {}, Oja {}",
                set.weights, set.size, set.hebb.stable, set.oja.stable
            );
        }
        match search.best() {
            Some(best) => println!("best         |S_u| = {}", best.size),
            None => println!("best         none of the checked sets is stable with both rules"),
        }
        println!("time         {:.2} s", start.elapsed().as_secs_f64());
        println!("written to   {}", out.path().display());
        Ok(())
    }
}
