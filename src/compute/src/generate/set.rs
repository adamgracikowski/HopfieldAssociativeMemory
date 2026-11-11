//! Generated pattern sets and their files.

use std::path::PathBuf;

use super::{GroupsConfig, GroupsOutput, GroupsProvenance};
use crate::patterns::npy::write_npy;
use crate::{Error, PatternSet, Result};

/// Pattern set generated for one output.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupSet {
    /// Output the set belongs to, with its file resolved against the
    /// directory of the configuration.
    pub output: GroupsOutput,
    /// Number of flipped pixels of every variant.
    pub flips: usize,
    /// The variants, group by group.
    pub set: PatternSet,
    /// Configuration of the generation.
    pub config: GroupsConfig,
}

impl GroupSet {
    /// Writes the set to the output `.npy` file, creating its directory,
    /// and its [`GroupsProvenance`] next to it, with the extension `.json`.
    pub fn write(&self) -> Result<()> {
        let file = &self.output.file;
        if let Some(parent) = file.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.to_owned(),
                source,
            })?;
        }
        write_npy(&self.set, file)?;
        let path = self.provenance_file();
        let io_error = |source: std::io::Error| Error::Io {
            path: path.clone(),
            source,
        };
        let mut text = serde_json::to_string_pretty(&GroupsProvenance::of(self))
            .map_err(|error| io_error(error.into()))?;
        text.push('\n');
        std::fs::write(&path, text).map_err(io_error)
    }

    /// File of the provenance: the output file with the extension `.json`.
    pub fn provenance_file(&self) -> PathBuf {
        self.output.file.with_extension("json")
    }
}
