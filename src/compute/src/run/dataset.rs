//! Dataset of a run.

use std::path::Path;

use serde::Serialize;

use crate::paths::relative_to;
use crate::patterns::registry::Registry;
use crate::{Error, PatternSet, Result};

/// The dataset of a run as recorded in its metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DatasetInfo {
    /// Name in the registry.
    pub name: String,
    /// Number of leading patterns of the registered set the run uses; `None`
    /// for all of them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix: Option<usize>,
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
    /// Number of patterns `p` (of the prefix, if any).
    pub patterns: usize,
    /// Number of neurons `N`.
    pub neurons: usize,
}

impl DatasetInfo {
    /// Loads dataset `name` from `registry` for a configuration file in the
    /// directory `config_dir`, against which the registry path is recorded;
    /// with `prefix`, only its first `prefix` patterns, which must be at
    /// least 1 and at most the size of the set.
    pub fn load(
        registry: &Registry,
        name: &str,
        prefix: Option<usize>,
        config_dir: &Path,
    ) -> Result<(Self, PatternSet)> {
        let entry = registry.entry(name)?;
        let mut set = registry.load(name)?;
        if let Some(patterns) = prefix {
            set = set.prefix(patterns).map_err(|_| Error::InvalidArgument {
                message: format!(
                    "cannot take the first {patterns} patterns of `{name}`, which has {}",
                    set.len()
                ),
            })?;
        }
        let info = Self {
            name: name.to_owned(),
            prefix,
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

    /// Name of the dataset in results: the registry name, followed by `:p`
    /// for a prefix of `p` patterns (e.g. `large-25x25.plus:6`).
    pub fn label(&self) -> String {
        match self.prefix {
            Some(patterns) => format!("{}:{patterns}", self.name),
            None => self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(prefix: Option<usize>) -> Result<(DatasetInfo, PatternSet)> {
        let registry = Registry::open(None, Path::new("../../data/experiments/x")).unwrap();
        DatasetInfo::load(&registry, "small-7x7", prefix, Path::new("."))
    }

    #[test]
    fn loads_a_prefix() {
        let (full, set) = load(None).unwrap();
        assert_eq!((full.label(), full.patterns), ("small-7x7".to_owned(), 5));
        let json = serde_json::to_value(&full).unwrap();
        assert!(json.get("prefix").is_none());

        let (info, prefix) = load(Some(3)).unwrap();
        assert_eq!((info.label(), info.patterns), ("small-7x7:3".to_owned(), 3));
        assert_eq!(prefix, set.prefix(3).unwrap());
        assert_eq!(serde_json::to_value(&info).unwrap()["prefix"], 3);
        for patterns in [0, 6] {
            let error = load(Some(patterns)).unwrap_err();
            assert!(
                error.to_string().contains(&format!(
                    "first {patterns} patterns of `small-7x7`, which has 5"
                )),
                "{error}"
            );
        }
    }
}
