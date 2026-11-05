//! Dataset of a run.

use std::path::Path;

use serde::Serialize;

use crate::paths::relative_to;
use crate::patterns::registry::Registry;
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
    /// Loads dataset `name` from `registry` for a configuration file in the
    /// directory `config_dir`, against which the registry path is recorded.
    pub fn load(registry: &Registry, name: &str, config_dir: &Path) -> Result<(Self, PatternSet)> {
        let entry = registry.entry(name)?;
        let set = registry.load(name)?;
        let info = Self {
            name: name.to_owned(),
            registry: relative_to(registry.path(), config_dir)
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
