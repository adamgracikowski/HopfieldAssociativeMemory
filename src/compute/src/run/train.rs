//! The `train` command: learn the weights of a dataset and save them.

use serde::Serialize;

use super::{DatasetInfo, Metadata, OutputDir, RunConfig, TrainingInfo};
use crate::dynamics::Stability;
use crate::learning::{LearningRule, Trained};
use crate::weights::Weights;
use crate::{PatternSet, Result};

/// Name of the dense weight file (`f64`, `N x N`).
pub const WEIGHTS_FILE: &str = "weights.npy";
/// Name of the coefficient file of factored weights (`f64`, `N x p`).
pub const COEFFICIENTS_FILE: &str = "coefficients.npy";
/// Name of the metadata file.
pub const METADATA_FILE: &str = "metadata.json";

/// A trained network with the stability of its patterns.
#[derive(Debug, Clone)]
pub struct TrainRun {
    /// The dataset.
    pub dataset: DatasetInfo,
    /// Its patterns.
    pub set: PatternSet,
    /// The trained weights.
    pub trained: Trained,
    /// Stability of the patterns, with `sgn(0)` from the dynamics settings.
    pub stability: Stability,
}

#[derive(Serialize)]
struct TrainResult<'a> {
    stability: &'a Stability,
}

impl TrainRun {
    /// Loads the dataset of `config`, trains the network and checks which
    /// patterns are stable.
    pub fn execute(config: &RunConfig) -> Result<Self> {
        let (dataset, set) = DatasetInfo::load(config)?;
        let trained = config.learning.train(&set, config.backend)?;
        let stability = Stability::of(&trained.weights, &set, config.dynamics.zero_field);
        Ok(Self {
            dataset,
            set,
            trained,
            stability,
        })
    }

    /// Writes `metadata.json` and the weights: [`WEIGHTS_FILE`] for dense
    /// weights, [`COEFFICIENTS_FILE`] (the matrix `A` of `W = A Xi - diag(d)`,
    /// with `Xi` the dataset) for factored ones.
    pub fn write(&self, config: &RunConfig, out: &OutputDir) -> Result<()> {
        let metadata = Metadata::new(
            "train",
            config,
            &self.dataset,
            TrainingInfo::from(&self.trained),
            TrainResult {
                stability: &self.stability,
            },
        );
        out.write_json(METADATA_FILE, &metadata)?;
        match &self.trained.weights {
            Weights::Dense(weights) => {
                let matrix = weights.matrix();
                out.write_npy(
                    WEIGHTS_FILE,
                    &[matrix.nrows(), matrix.ncols()],
                    matrix.iter().copied(),
                )
            }
            Weights::Factored(weights) => {
                let coefficients = weights.coefficients();
                out.write_npy(
                    COEFFICIENTS_FILE,
                    &[coefficients.nrows(), coefficients.ncols()],
                    coefficients.iter().copied(),
                )
            }
        }
    }
}
