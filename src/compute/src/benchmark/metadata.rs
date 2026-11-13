//! Metadata file of a benchmark.

use serde::Serialize;

use super::{BenchmarkConfig, BenchmarkRun};
use crate::build_info::BuildInfo;
use crate::run::DatasetInfo;
use crate::run::metadata::timestamp;

/// Contents of the `metadata.json` of a benchmark: what was run, with which
/// code, settings and number of threads.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkMetadata<'a> {
    /// Subcommand, `benchmark`.
    pub command: &'static str,
    /// Time of the run (RFC 3339, UTC, seconds).
    pub created: String,
    /// Build of the binary.
    pub build: BuildInfo,
    /// Configuration with every default filled in.
    pub config: &'a BenchmarkConfig,
    /// The registered datasets, in configuration order.
    pub datasets: &'a [DatasetInfo],
    /// Number of threads of the parallel local fields.
    pub threads: usize,
    /// Whether the heap was measured.
    pub heap_measured: bool,
}

impl<'a> BenchmarkMetadata<'a> {
    /// Metadata of `run`, executed now with the current build from `config`.
    pub fn new(config: &'a BenchmarkConfig, run: &'a BenchmarkRun) -> Self {
        Self {
            command: "benchmark",
            created: timestamp(),
            build: BuildInfo::CURRENT,
            config,
            datasets: &run.datasets,
            threads: run.threads,
            heap_measured: run.heap_measured,
        }
    }
}
