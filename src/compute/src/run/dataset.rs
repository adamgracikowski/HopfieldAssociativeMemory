//! Dataset of a run.

use serde::Serialize;

use super::RunConfig;
use crate::paths::relative_to;
use crate::{PatternSet, Result};

/// The dataset of a run as recorded in its metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DatasetInfo {
    /// Name in the registry.
    pub name: String,
    /// Registry file, relative to the directory of the configuration (with
    /// `/` separators), or as found when no relative path exists.
    pub registry: String,
    /// Pattern file as written in the registry, relative to its directory.
    pub file: String,
    /// Description from the registry.
    pub description: String,
    /// Bitmap width.
    pub width: usize,
    /// Bitmap height.
    pub height: usize,
    /// Number of patterns `p`.
    pub patterns: usize,
    /// Number of neurons `N`.
    pub neurons: usize,
}

impl DatasetInfo {
    /// Loads the dataset of `config` from its registry.
    pub fn load(config: &RunConfig) -> Result<(Self, PatternSet)> {
        let registry = config.open_registry()?;
        let entry = registry.entry(&config.dataset)?;
        let set = registry.load(&config.dataset)?;
        let info = Self {
            name: config.dataset.clone(),
            registry: relative_to(registry.path(), config.directory())
                .unwrap_or_else(|| registry.path().display().to_string()),
            file: entry.file.to_string_lossy().replace('\\', "/"),
            description: entry.description.clone(),
            width: set.width(),
            height: set.height(),
            patterns: set.len(),
            neurons: set.neurons(),
        };
        Ok((info, set))
    }
}
