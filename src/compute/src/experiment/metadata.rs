//! Metadata file of an experiment.

use serde::Serialize;

use super::{ExperimentConfig, ExperimentRun, Training};
use crate::build_info::BuildInfo;
use crate::run::DatasetInfo;
use crate::run::metadata::timestamp;

/// Contents of the `metadata.json` of an experiment: what was run, with
/// which code and settings, and the training of every (dataset, rule) pair.
/// Running the experiment again changes only `created`.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentMetadata<'a> {
    /// Subcommand, `experiment`.
    pub command: &'static str,
    /// Time of the run (RFC 3339, UTC, seconds).
    pub created: String,
    /// Build of the binary.
    pub build: BuildInfo,
    /// Configuration with every default filled in.
    pub config: &'a ExperimentConfig,
    /// The datasets in configuration order.
    pub datasets: &'a [DatasetInfo],
    /// Trainings ordered by dataset, then rule.
    pub trainings: &'a [Training],
}

impl<'a> ExperimentMetadata<'a> {
    /// Metadata of `run`, executed now with the current build from `config`.
    pub fn new(config: &'a ExperimentConfig, run: &'a ExperimentRun) -> Self {
        Self {
            command: "experiment",
            created: timestamp(),
            build: BuildInfo::CURRENT,
            config,
            datasets: &run.datasets,
            trainings: &run.trainings,
        }
    }
}
