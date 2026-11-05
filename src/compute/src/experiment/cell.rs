//! Cells of the experiment grid.

use serde::Serialize;

use crate::dynamics::Mode;

/// One combination of the grid axes of an experiment. Datasets and learning
/// rules are given by their position in the configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Cell {
    /// Index into `datasets`.
    pub dataset: usize,
    /// Index into `learning`.
    pub learning: usize,
    /// Update mode.
    pub mode: Mode,
    /// Fraction of flipped pixels; `None` for random initial states.
    pub noise: Option<f64>,
}
