//! Result of a simulation.

use super::Termination;
use crate::patterns::Pattern;

/// Final state of a simulation and how it was reached.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// State after `steps` steps. For a [`Termination::TwoCycle`] it is one
    /// of the two alternating states, the other being the state one step
    /// earlier.
    pub state: Pattern,
    /// Number of steps applied (synchronous updates or asynchronous sweeps).
    /// The step that only confirms a fixed point or reveals a cycle is not
    /// applied, so a stable initial state gives 0.
    pub steps: usize,
    /// Total number of neuron state changes over all steps.
    pub changes: usize,
    /// Why the simulation stopped.
    pub termination: Termination,
    /// Energy `E = -1/2 x^T W x` of the final state.
    pub energy: f64,
}
