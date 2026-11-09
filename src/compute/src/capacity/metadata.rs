//! Metadata file of a capacity run.

use serde::Serialize;

use super::{CapacityConfig, CapacityRun};
use crate::build_info::BuildInfo;
use crate::run::DatasetInfo;
use crate::run::metadata::timestamp;

/// Contents of the `metadata.json` of a capacity run: what was run, with
/// which code and settings. Running it again changes only `created`.
#[derive(Debug, Clone, Serialize)]
pub struct CapacityMetadata<'a> {
    /// Subcommand, `capacity`.
    pub command: &'static str,
    /// Time of the run (RFC 3339, UTC, seconds).
    pub created: String,
    /// Build of the binary.
    pub build: BuildInfo,
    /// Configuration with every default filled in.
    pub config: &'a CapacityConfig,
    /// The datasets whose prefixes were checked, in configuration order.
    pub datasets: &'a [DatasetInfo],
}

impl<'a> CapacityMetadata<'a> {
    /// Metadata of `run`, executed now with the current build from `config`.
    pub fn new(config: &'a CapacityConfig, run: &'a CapacityRun) -> Self {
        Self {
            command: "capacity",
            created: timestamp(),
            build: BuildInfo::CURRENT,
            config,
            datasets: &run.datasets,
        }
    }
}
