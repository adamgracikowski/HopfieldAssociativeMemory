//! Cells of the experiment grid.

use serde::Serialize;

use crate::dynamics::Mode;

/// One combination of the grid axes of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Cell {
    /// Index into `datasets`.
    pub dataset: usize,
    /// Index into `learning`.
    pub learning: usize,
    /// Update mode.
    pub mode: Mode,
    /// Damage level: the fraction of flipped pixels, or of covered rows;
    /// `None` for random initial states.
    pub noise: Option<f64>,
}

impl Cell {
    /// Whether every trial gives the same runs: synchronous updates of
    /// undamaged patterns (`pixels = 0`).
    pub fn is_deterministic(&self, pixels: usize) -> bool {
        self.mode == Mode::Synchronous && self.noise.is_some() && pixels == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_undamaged_synchronous_cells_are_deterministic() {
        let cell = |mode, noise| Cell {
            dataset: 0,
            learning: 0,
            mode,
            noise,
        };
        assert!(cell(Mode::Synchronous, Some(0.0)).is_deterministic(0));
        assert!(cell(Mode::Synchronous, Some(0.001)).is_deterministic(0));
        assert!(!cell(Mode::Synchronous, Some(0.1)).is_deterministic(5));
        assert!(!cell(Mode::Asynchronous, Some(0.0)).is_deterministic(0));
        assert!(!cell(Mode::Synchronous, None).is_deterministic(0));
    }
}
