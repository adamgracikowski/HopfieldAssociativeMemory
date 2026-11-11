//! Entries of the dataset axis of an experiment.

use std::fmt;

use serde::{Deserialize, Serialize};

/// One value of the `datasets` axis: a registry name, or a table with the
/// name and the number of leading patterns to store:
///
/// ```toml
/// datasets = ["small-7x7", { name = "large-25x25.plus", patterns = 6 }]
/// ```
///
/// [`Display`](fmt::Display) gives its label in the results: the name, or
/// `name:p` for a prefix (e.g. `large-25x25.plus:6`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum DatasetEntry {
    /// Every pattern of the dataset.
    Name(String),
    /// The first patterns of the dataset.
    Prefix(DatasetPrefix),
}

/// The first [`patterns`](Self::patterns) patterns of dataset
/// [`name`](Self::name).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetPrefix {
    /// Name in the registry.
    pub name: String,
    /// Number of leading patterns, from 1.
    pub patterns: usize,
}

impl DatasetEntry {
    /// Name in the registry.
    pub fn name(&self) -> &str {
        match self {
            Self::Name(name) | Self::Prefix(DatasetPrefix { name, .. }) => name,
        }
    }

    /// Number of leading patterns, or `None` for the whole dataset.
    pub fn prefix(&self) -> Option<usize> {
        match self {
            Self::Name(_) => None,
            Self::Prefix(prefix) => Some(prefix.patterns),
        }
    }
}

impl From<&str> for DatasetEntry {
    fn from(name: &str) -> Self {
        Self::Name(name.to_owned())
    }
}

impl fmt::Display for DatasetEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(name) => f.write_str(name),
            Self::Prefix(DatasetPrefix { name, patterns }) => write!(f, "{name}:{patterns}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, Serialize)]
    struct Wrapper {
        datasets: Vec<DatasetEntry>,
    }

    #[test]
    fn parses_names_and_prefixes() {
        let text = "datasets = [\"small-7x7\", { name = \"large-25x25.plus\", patterns = 6 }]";
        let entries = toml::from_str::<Wrapper>(text).unwrap().datasets;
        assert_eq!(entries[0], DatasetEntry::from("small-7x7"));
        assert_eq!(
            (entries[0].name(), entries[0].prefix()),
            ("small-7x7", None)
        );
        assert_eq!(entries[0].to_string(), "small-7x7");
        assert_eq!(
            (entries[1].name(), entries[1].prefix()),
            ("large-25x25.plus", Some(6))
        );
        assert_eq!(entries[1].to_string(), "large-25x25.plus:6");

        let json = serde_json::to_value(Wrapper { datasets: entries }).unwrap();
        assert_eq!(json["datasets"][0], "small-7x7");
        assert_eq!(json["datasets"][1]["patterns"], 6);

        for bad in [
            "datasets = [{ name = \"a\" }]",
            "datasets = [{ name = \"a\", patterns = 2, extra = 1 }]",
            "datasets = [3]",
        ] {
            assert!(toml::from_str::<Wrapper>(bad).is_err(), "{bad}");
        }
    }
}
