//! Experiments: a grid of datasets, learning rules, update modes and noise
//! levels, with many trials in every cell.

pub mod cell;
pub mod config;
pub mod dynamics;
pub mod input;

pub use cell::Cell;
pub use config::ExperimentConfig;
pub use dynamics::ExperimentDynamics;
pub use input::ExperimentInput;
