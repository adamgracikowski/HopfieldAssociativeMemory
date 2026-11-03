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
    /// Weights storing the patterns of `set`, kept in `backend`
    /// ([`Backend::Auto`] is resolved by the number of neurons), with the
    /// number of epochs and whether learning converged.
    fn train(&self, set: &PatternSet, backend: Backend) -> Result<Trained>;
}
