//! Result of Oja's rule.

use ndarray::Array2;

/// Result of Oja's rule before it is stored in a weight backend.
#[derive(Debug, Clone, PartialEq)]
pub struct OjaFit {
    /// Coefficients `A` (`N x p`) of `W = A Xi`, diagonal not yet removed.
    pub coefficients: Array2<f64>,
    /// Number of epochs run.
    pub epochs: usize,
    /// Whether the change fell below the tolerance before the epoch limit.
    pub converged: bool,
}
