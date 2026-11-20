//! Learning rules that compute the weights of a Hopfield network from a
//! pattern set.

pub mod hebb;
pub mod oja;
pub mod trained;

pub use hebb::Hebb;
pub use oja::{Oja, OjaFit, OjaVariant};
pub use trained::Trained;

use crate::weights::Backend;
use crate::{PatternSet, Result};

/// A rule that computes the weights storing the patterns of a set.
pub trait LearningRule {
    /// Weights storing the patterns of `set`, kept in `backend`.
    fn train(&self, set: &PatternSet, backend: Backend) -> Result<Trained>;
}
