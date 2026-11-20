//! Origin of a converted dataset.

use std::path::Path;

use serde::Serialize;

use super::{Conversion, Source};
use crate::build_info::BuildInfo;
use crate::images::Binarization;
use crate::paths::relative_to;

/// Origin of a converted dataset, written next to its `.npy` file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Provenance<'a> {
    /// Command that produced the dataset.
    pub command: &'static str,
    /// Build of the binary.
    pub build: BuildInfo,
    /// Configuration file, relative to the directory of the dataset (with
    /// `/` separators), or as given when no relative path exists.
    pub config: String,
    /// Bitmap width.
    pub width: usize,
    /// Bitmap height.
    pub height: usize,
    /// Binarization method.
    pub method: Binarization,
    /// Source images, one per pattern.
    pub sources: &'a [Source],
}

impl<'a> Provenance<'a> {
    /// Provenance of `conversion` with the current build.
    pub fn of(conversion: &'a Conversion) -> Self {
        let dir = conversion.output.file.parent().unwrap_or(Path::new(""));
        Self {
            command: "convert-images",
            build: BuildInfo::CURRENT,
            config: relative_to(&conversion.config, dir)
                .unwrap_or_else(|| conversion.config.display().to_string()),
            width: conversion.set.width(),
            height: conversion.set.height(),
            method: conversion.output.method,
            sources: &conversion.sources,
        }
    }
}
