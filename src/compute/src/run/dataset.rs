//! Dataset of a run.

use std::path::PathBuf;

use serde::Serialize;

use super::RunConfig;
use crate::{PatternSet, Result};

/// The dataset of a run as recorded in its metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DatasetInfo {
    /// Name in the registry.
    pub name: String,
    /// Pattern file.
    pub file: PathBuf,
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
            file: registry.file_path(&config.dataset)?,
            description: entry.description.clone(),
            width: set.width(),
            height: set.height(),
            patterns: set.len(),
            neurons: set.neurons(),
        };
        Ok((info, set))
    }
}
