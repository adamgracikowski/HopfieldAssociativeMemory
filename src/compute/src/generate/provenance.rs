//! Origin of a generated dataset.

use std::path::Path;

use serde::Serialize;

use super::GroupSet;
use crate::build_info::BuildInfo;
use crate::paths::relative_to;

/// Origin of a generated dataset, written next to its `.npy` file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupsProvenance<'a> {
    /// Command that produced the dataset.
    pub command: &'static str,
    /// Build of the binary.
    pub build: BuildInfo,
    /// Configuration file, relative to the directory of the dataset (with
    /// `/` separators), or as given when no relative path exists.
    pub config: String,
    /// Master seed of the variants.
    pub seed: u64,
    /// Registered dataset with the prototypes.
    pub dataset: &'a str,
    /// Indices of the prototypes, one group each, in pattern order.
    pub prototypes: &'a [usize],
    /// Number of variants of every prototype.
    pub variants: usize,
    /// Fraction of flipped pixels of every variant.
    pub flip: f64,
    /// Number of flipped pixels of every variant.
    pub flips: usize,
    /// Bitmap width.
    pub width: usize,
    /// Bitmap height.
    pub height: usize,
}

impl<'a> GroupsProvenance<'a> {
    /// Provenance of `generated` with the current build.
    pub fn of(generated: &'a GroupSet) -> Self {
        let dir = generated.output.file.parent().unwrap_or(Path::new(""));
        let config = &generated.config;
        Self {
            command: "generate",
            build: BuildInfo::CURRENT,
            config: relative_to(config.path(), dir)
                .unwrap_or_else(|| config.path().display().to_string()),
            seed: config.seed,
            dataset: &config.dataset,
            prototypes: &config.prototypes,
            variants: config.variants,
            flip: generated.output.flip,
            flips: generated.flips,
            width: generated.set.width(),
            height: generated.set.height(),
        }
    }
}
