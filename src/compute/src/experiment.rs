//! Experiments: a grid of datasets, learning rules, update modes and damage
//! levels, with many trials in every cell.

pub mod attractor;
pub mod attractor_table;
pub mod cell;
pub mod class;
pub mod config;
pub mod damage;
pub mod dataset;
pub mod dynamics;
pub mod fields;
pub mod input;
pub mod metadata;
pub mod mixture;
pub mod runner;
pub mod statistics;
pub mod summary;
pub mod table;
pub mod trial;

pub use attractor::Attractor;
pub use attractor_table::AttractorTable;
pub use cell::Cell;
pub use class::FinalClass;
pub use config::ExperimentConfig;
pub use damage::Damage;
pub use dataset::{DatasetEntry, DatasetPrefix};
pub use dynamics::ExperimentDynamics;
pub use fields::FieldTable;
pub use input::ExperimentInput;
pub use metadata::ExperimentMetadata;
pub use mixture::{MIXTURE_SIZES, Mixture};
pub use runner::{ExperimentRun, Training};
pub use statistics::{DatasetStatistics, MaxOverlap};
pub use summary::{CellSummary, Classes, Moments, Proportion};
pub use table::TrialTable;
pub use trial::{Trial, TrialResult};
