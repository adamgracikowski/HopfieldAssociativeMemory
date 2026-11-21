//! Result of a learning rule.

use crate::weights::Weights;

/// Weights produced by a learning rule with a summary of the learning.
#[derive(Debug, Clone, PartialEq)]
pub struct Trained {
    /// Learned weights.
    pub weights: Weights,
    /// Number of epochs of an iterative rule; 0 for rules computed in closed
    /// form, such as the Hebbian rule.
    pub epochs: usize,
    /// Whether an iterative rule met its stopping criterion before the epoch
    /// limit; always `true` for rules computed in closed form.
    pub converged: bool,
}
