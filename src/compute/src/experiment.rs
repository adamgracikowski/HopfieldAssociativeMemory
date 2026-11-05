//! Experiments: a grid of datasets, learning rules, update modes and noise
//! levels, with many trials in every cell.

pub mod cell;
pub mod class;
pub mod config;
pub mod dynamics;
pub mod input;
pub mod metadata;
pub mod runner;
pub mod summary;
pub mod table;
pub mod trial;

pub use cell::Cell;
pub use class::FinalClass;
pub use config::ExperimentConfig;
pub use dynamics::ExperimentDynamics;
pub use input::ExperimentInput;
pub use metadata::ExperimentMetadata;
pub use runner::{ExperimentRun, Training};
pub use summary::{CellSummary, Classes, Moments, Proportion};
pub use table::TrialTable;
pub use trial::{Trial, TrialResult};
