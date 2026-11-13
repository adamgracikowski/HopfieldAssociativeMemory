//! The `benchmark` subcommand.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use compute::benchmark::{BenchmarkConfig, BenchmarkRun, HeapProbe};
use compute::run::OutputDir;

/// Directory of the results next to the configuration file.
const RESULTS_DIR: &str = "results";

#[derive(Debug, Args)]
pub struct BenchmarkArgs {
    /// Benchmark configuration (TOML) with the seed, the datasets, the
    /// random sizes, the backends and the learning rules.
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

/// The heap as counted by the global allocator of the binary.
struct AllocatorProbe;

impl HeapProbe for AllocatorProbe {
    fn start(&self) -> Option<usize> {
        crate::HEAP.reset_peak_usage();
        Some(crate::HEAP.current_usage())
    }

    fn peak(&self) -> Option<usize> {
        Some(crate::HEAP.peak_usage())
    }
}

impl BenchmarkArgs {
    /// Measures every configured set, backend and rule, writes the table
    /// and prints one line per measurement and a summary to stdout.
    pub fn run(self) -> compute::Result<()> {
        let start = Instant::now();
        let config = BenchmarkConfig::from_file(&self.config)?;
        let path = self
            .out
            .unwrap_or_else(|| config.directory().join(RESULTS_DIR));
        let out = OutputDir::new(&path, self.force)?;
        let labels = config.learning_labels();
        let run = BenchmarkRun::execute(&config, &AllocatorProbe, |set, m| {
            println!(
                "{:<20} N = {:<6} {:<8} {:<15} #{}  train {:.4} s  sync {:.4} s  async {:.4} s",
                set.source,
                set.neurons,
                m.backend,
                labels[m.learning],
                m.repeat,
                m.train_seconds,
                m.sync_seconds,
                m.async_seconds
            );
        })?;
        run.write(&config, &out)?;

        println!("measurements {}", run.measurements.len());
        println!("time         {:.2} s", start.elapsed().as_secs_f64());
        println!("written to   {}", out.path().display());
        Ok(())
    }
}
