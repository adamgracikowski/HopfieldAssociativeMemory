//! Metadata file of a run.

use std::time::SystemTime;

use serde::Serialize;

use super::{DatasetInfo, RunConfig};
use crate::build_info::BuildInfo;
use crate::learning::Trained;
use crate::weights::{Backend, WeightMatrix};

/// The current time in RFC 3339 format, UTC, to the second, as recorded in
/// the `created` field of metadata files.
pub fn timestamp() -> String {
    humantime::format_rfc3339_seconds(SystemTime::now()).to_string()
}

/// Result of training as recorded in the metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TrainingInfo {
    /// Backend that stores the weights (`auto` resolved).
    pub backend: Backend,
    /// Number of neurons.
    pub neurons: usize,
    /// Epochs of an iterative rule, 0 for a closed form.
    pub epochs: usize,
    /// Whether the rule converged.
    pub converged: bool,
}

impl From<&Trained> for TrainingInfo {
    fn from(trained: &Trained) -> Self {
        Self {
            backend: trained.weights.backend(),
            neurons: trained.weights.neurons(),
            epochs: trained.epochs,
            converged: trained.converged,
        }
    }
}

/// Contents of `metadata.json`: what was run, with which code and settings,
/// and the command-specific `result`, whose fields are written at the top
/// level.
#[derive(Debug, Clone, Serialize)]
pub struct Metadata<'a, R: Serialize> {
    /// Subcommand (`train`, `recall`).
    pub command: &'static str,
    /// Time of the run (RFC 3339, UTC, seconds).
    pub created: String,
    /// Build of the binary.
    pub build: BuildInfo,
    /// Configuration with every default filled in.
    pub config: &'a RunConfig,
    /// The dataset.
    pub dataset: &'a DatasetInfo,
    /// Result of training.
    pub training: TrainingInfo,
    /// Command-specific results.
    #[serde(flatten)]
    pub result: R,
}

impl<'a, R: Serialize> Metadata<'a, R> {
    /// Metadata of `command` run now with the current build.
    pub fn new(
        command: &'static str,
        config: &'a RunConfig,
        dataset: &'a DatasetInfo,
        training: TrainingInfo,
        result: R,
    ) -> Self {
        Self {
            command,
            created: timestamp(),
            build: BuildInfo::CURRENT,
            config,
            dataset,
            training,
            result,
        }
    }
}
