//! Origin of a dataset written by a search for stable sets.

use std::path::Path;

use serde::Serialize;

use super::StableConfig;
use crate::PatternSet;
use crate::build_info::BuildInfo;
use crate::paths::relative_to;
use crate::patterns::npy::write_npy;
use crate::{Error, Result};

/// Where a dataset of a search comes from, written next to its `.npy` file:
/// the build, the configuration, the hyperplane and what part of its set
/// the file holds. It has no timestamp, so searching again with the same
/// code and configuration gives the same file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StableProvenance<'a> {
    /// Command that produced the dataset.
    pub command: &'static str,
    /// Build of the binary.
    pub build: BuildInfo,
    /// Configuration file, relative to the directory of the dataset (with
    /// `/` separators), or as given when no relative path exists.
    pub config: String,
    /// Weights `u` of the hyperplane.
    pub weights: &'a [u32],
    /// Size of the whole set `S_u`.
    pub set_size: u64,
    /// What the file holds: `basis` or `sample`.
    pub content: &'static str,
    /// Number of patterns in the file.
    pub patterns: usize,
    /// Bitmap width.
    pub width: usize,
    /// Bitmap height.
    pub height: usize,
}

impl<'a> StableProvenance<'a> {
    /// Provenance of `set`, part `content` of `S_u` with `weights` and
    /// `set_size` members, written to `file`.
    pub fn new(
        config: &StableConfig,
        file: &Path,
        weights: &'a [u32],
        set_size: u64,
        content: &'static str,
        set: &PatternSet,
    ) -> Self {
        let dir = file.parent().unwrap_or(Path::new(""));
        Self {
            command: "search-stable",
            build: BuildInfo::CURRENT,
            config: relative_to(config.path(), dir)
                .unwrap_or_else(|| config.path().display().to_string()),
            weights,
            set_size,
            content,
            patterns: set.len(),
            width: set.width(),
            height: set.height(),
        }
    }

    /// Writes `set` to `file`, creating its directory, and this provenance
    /// next to it, with the extension `.json`.
    pub fn write(&self, set: &PatternSet, file: &Path) -> Result<()> {
        if let Some(parent) = file.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.to_owned(),
                source,
            })?;
        }
        write_npy(set, file)?;
        let path = file.with_extension("json");
        let io_error = |source: std::io::Error| Error::Io {
            path: path.clone(),
            source,
        };
        let mut text =
            serde_json::to_string_pretty(self).map_err(|error| io_error(error.into()))?;
        text.push('\n');
        std::fs::write(&path, text).map_err(io_error)
    }
}
