//! Observers of the states visited by a simulation.

use crate::patterns::PatternView;

/// Receives the frames of a simulation: the initial state, the states
/// between steps and the final state.
pub trait Recorder {
    /// Records `state` with its energy and the number of neurons that
    /// changed since the previous frame (0 for the first frame).
    fn record(&mut self, state: PatternView<'_>, energy: f64, changes: usize);
}

/// Recorder that keeps nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoHistory;

impl Recorder for NoHistory {
    fn record(&mut self, _state: PatternView<'_>, _energy: f64, _changes: usize) {}
}
