//! Result of a simulation.

use super::Termination;
use crate::patterns::Pattern;

/// Final state of a simulation and how it was reached.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// State after `steps` steps; for a 2-cycle one of its two states.
    pub state: Pattern,
    /// Number of applied steps (synchronous updates or asynchronous sweeps);
    /// 0 for a stable initial state.
    pub steps: usize,
    /// Total number of neuron state changes over all steps.
    pub changes: usize,
    /// Why the simulation stopped.
    pub termination: Termination,
    /// Energy `E = -1/2 x^T W x` of the final state.
    pub energy: f64,
}
