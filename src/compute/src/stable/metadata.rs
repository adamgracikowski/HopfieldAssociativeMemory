//! Metadata file of a search for stable sets.

use serde::Serialize;

use super::StableConfig;
use crate::build_info::BuildInfo;
use crate::run::metadata::timestamp;

/// Contents of the `metadata.json` of a search: what was run, with which
/// code and settings.
#[derive(Debug, Clone, Serialize)]
pub struct StableMetadata<'a> {
    /// Subcommand, `search-stable`.
    pub command: &'static str,
    /// Time of the run (RFC 3339, UTC, seconds).
    pub created: String,
    /// Build of the binary.
    pub build: BuildInfo,
    /// Configuration with every default filled in.
    pub config: &'a StableConfig,
}

impl<'a> StableMetadata<'a> {
    /// Metadata of a search run now with the current build from `config`.
    pub fn new(config: &'a StableConfig) -> Self {
        Self {
            command: "search-stable",
            created: timestamp(),
            build: BuildInfo::CURRENT,
            config,
        }
    }
}
