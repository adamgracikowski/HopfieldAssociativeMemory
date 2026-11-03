//! Registry of pattern sets (`data/datasets/datasets.toml`): file, bitmap
//! size and description of every dataset.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::PatternSet;
use super::csv::read_csv;
use super::npy::read_npy;
use crate::{Error, Result};

/// Registry location relative to the repository root.
pub const DEFAULT_REGISTRY: &str = "data/datasets/datasets.toml";

/// One registered dataset.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetEntry {
    /// Pattern file, relative to the registry directory.
    pub file: PathBuf,
    /// Bitmap width in pixels.
    pub width: usize,
    /// Bitmap height in pixels.
    pub height: usize,
    /// Field separator of a CSV file (one ASCII character, `,` by default).
    #[serde(default = "default_separator")]
    pub separator: char,
    /// What the patterns show.
    pub description: String,
}

fn default_separator() -> char {
    ','
}

impl DatasetEntry {
    /// Separator as the byte expected by the CSV reader.
    fn separator_byte(&self) -> std::result::Result<u8, String> {
        match self.separator {
            '"' | '\n' | '\r' | '-' | '0'..='9' => Err(format!(
                "separator `{}` is not allowed",
                self.separator.escape_default()
            )),
            c if c.is_ascii() => Ok(c as u8),
            c => Err(format!("separator `{c}` is not an ASCII character")),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    datasets: BTreeMap<String, DatasetEntry>,
}

/// Datasets of a registry file, sorted by name.
#[derive(Debug, Clone)]
pub struct Registry {
    path: PathBuf,
    datasets: BTreeMap<String, DatasetEntry>,
}

impl Registry {
    /// Reads a registry file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parses registry text and checks every entry. `path` is the registry
    /// location: pattern files are resolved relative to its directory.
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let file: RegistryFile = toml::from_str(text).map_err(|source| Error::Registry {
            path: path.to_owned(),
            source,
        })?;
        for (name, entry) in &file.datasets {
            entry
                .separator_byte()
                .map_err(|message| Error::InvalidEntry {
                    registry: path.to_owned(),
                    name: name.clone(),
                    message,
                })?;
        }
        Ok(Self {
            path: path.to_owned(),
            datasets: file.datasets,
        })
    }

    /// Finds [`DEFAULT_REGISTRY`] in `start` or the nearest of its parent
    /// directories and reads it.
    pub fn discover(start: &Path) -> Result<Self> {
        let relative = Path::new(DEFAULT_REGISTRY);
        start
            .ancestors()
            .map(|dir| dir.join(relative))
            .find(|candidate| candidate.is_file())
            .ok_or_else(|| Error::RegistryNotFound {
                start: start.to_owned(),
                relative: relative.to_owned(),
            })
            .and_then(|path| Self::from_file(&path))
    }

    /// Location of the registry file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Names of the registered datasets in alphabetical order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.datasets.keys().map(String::as_str)
    }

    /// Entry of dataset `name`.
    pub fn entry(&self, name: &str) -> Result<&DatasetEntry> {
        self.datasets
            .get(name)
            .ok_or_else(|| Error::UnknownDataset {
                name: name.to_owned(),
                registry: self.path.clone(),
                known: self.names().collect::<Vec<_>>().join(", "),
            })
    }

    /// Path of the pattern file of dataset `name`.
    pub fn file_path(&self, name: &str) -> Result<PathBuf> {
        let entry = self.entry(name)?;
        let dir = self.path.parent().unwrap_or_else(|| Path::new(""));
        Ok(dir.join(&entry.file))
    }

    /// Reads the patterns of dataset `name`.
    pub fn load(&self, name: &str) -> Result<PatternSet> {
        let entry = self.entry(name)?;
        let path = self.file_path(name)?;
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("csv") => {
                let separator = entry
                    .separator_byte()
                    .expect("separators are checked when the registry is parsed");
                read_csv(&path, entry.width, entry.height, separator)
            }
            Some("npy") => {
                let set = read_npy(&path)?;
                if (set.width(), set.height()) != (entry.width, entry.height) {
                    return Err(Error::Shape {
                        message: format!(
                            "`{}` holds {}x{} bitmaps, the registry declares {}x{}",
                            path.display(),
                            set.width(),
                            set.height(),
                            entry.width,
                            entry.height
                        ),
                    });
                }
                Ok(set)
            }
            _ => Err(Error::UnsupportedFormat { path }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = r#"
[datasets.b]
file = "b.csv"
width = 2
height = 3
description = "Second."

[datasets."a.plus"]
file = "sub/a.csv"
width = 1
height = 1
separator = ";"
description = "First."
"#;

    fn entry_with_separator(separator: &str) -> String {
        format!(
            "[datasets.a]\nfile = \"a.csv\"\nwidth = 1\nheight = 1\nseparator = \"{separator}\"\ndescription = \"\"\n"
        )
    }

    fn registry() -> Registry {
        Registry::parse(TEXT, Path::new("data/datasets.toml")).unwrap()
    }

    #[test]
    fn lists_names_in_order_and_resolves_files() {
        let registry = registry();
        assert_eq!(registry.names().collect::<Vec<_>>(), ["a.plus", "b"]);
        assert_eq!(registry.entry("b").unwrap().width, 2);
        assert_eq!(
            registry.file_path("a.plus").unwrap(),
            Path::new("data").join("sub/a.csv")
        );
    }

    #[test]
    fn separator_defaults_to_a_comma() {
        let registry = registry();
        assert_eq!(registry.entry("b").unwrap().separator, ',');
        assert_eq!(registry.entry("a.plus").unwrap().separator, ';');
    }

    #[test]
    fn rejects_invalid_separators() {
        for separator in ["-", "1", "ł", "\\\""] {
            let error =
                Registry::parse(&entry_with_separator(separator), Path::new("r.toml")).unwrap_err();
            assert!(
                matches!(&error, Error::InvalidEntry { name, .. } if name == "a"),
                "{separator}: {error}"
            );
        }
        assert!(Registry::parse(&entry_with_separator("\\t"), Path::new("r.toml")).is_ok());
    }

    #[test]
    fn unknown_dataset_lists_the_known_ones() {
        let error = registry().entry("c").unwrap_err();
        assert!(matches!(&error, Error::UnknownDataset { known, .. } if known == "a.plus, b"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let text =
            "[datasets.a]\nfile = \"a.csv\"\nwidth = 1\nheight = 1\ndescription = \"\"\nseed = 1\n";
        let error = Registry::parse(text, Path::new("r.toml")).unwrap_err();
        assert!(matches!(error, Error::Registry { .. }));
    }

    #[test]
    fn rejects_unsupported_formats() {
        let text = "[datasets.a]\nfile = \"a.txt\"\nwidth = 1\nheight = 1\ndescription = \"\"\n";
        let error = Registry::parse(text, Path::new("r.toml"))
            .unwrap()
            .load("a")
            .unwrap_err();
        assert!(matches!(error, Error::UnsupportedFormat { .. }));
    }
}
