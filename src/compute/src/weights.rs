//! Weight matrices of a Hopfield network.
//!
//! Two storage backends implement [`WeightMatrix`]:
//! - [`DenseWeights`]: the full `N x N` matrix, for small and medium networks,
//! - [`FactoredWeights`]: `W = A Xi - diag(d)` built from the `p` stored
//!   patterns, using `O(N p)` memory, for large bitmaps where the dense
//!   matrix would not fit in memory.
//!
//! [`Weights`] holds either of them and is chosen with [`Backend`].

pub mod backend;
pub mod dense;
pub mod factored;

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

pub use backend::Backend;
pub use dense::DenseWeights;
pub use factored::FactoredWeights;

/// A weight matrix `W` with zero diagonal, read through the local fields
/// `h = W x` of a bipolar state `x`.
pub trait WeightMatrix {
    /// Number of neurons `N`.
    fn neurons(&self) -> usize;

    /// Writes the local fields `h = W x` of `state` into `out`, without
    /// allocating an `N x N` temporary.
    ///
    /// # Panics
    ///
    /// Panics if `state` or `out` does not have `N` elements.
    fn local_fields_into(&self, state: ArrayView1<'_, i8>, out: ArrayViewMut1<'_, f64>);

    /// Local field `h_i = sum_j w_ij x_j` of neuron `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N` or `state` does not have `N` elements.
    fn local_field(&self, i: usize, state: ArrayView1<'_, i8>) -> f64;

    /// The full `N x N` matrix. Meant for tests and small networks: it
    /// allocates `N^2` values.
    fn to_dense(&self) -> Array2<f64>;

    /// Local fields `h = W x` of `state` as a new vector.
    ///
    /// # Panics
    ///
    /// Panics if `state` does not have `N` elements.
    fn local_fields(&self, state: ArrayView1<'_, i8>) -> Array1<f64> {
        let mut out = Array1::zeros(self.neurons());
        self.local_fields_into(state, out.view_mut());
        out
    }
}

/// Weights stored in one of the backends.
#[derive(Debug, Clone, PartialEq)]
pub enum Weights {
    /// Full `N x N` matrix.
    Dense(DenseWeights),
    /// Factored matrix built from the stored patterns.
    Factored(FactoredWeights),
}

impl Weights {
    /// Backend that stores these weights ([`Backend::Dense`] or
    /// [`Backend::Factored`]).
    pub fn backend(&self) -> Backend {
        match self {
            Self::Dense(_) => Backend::Dense,
            Self::Factored(_) => Backend::Factored,
        }
    }

    fn inner(&self) -> &dyn WeightMatrix {
        match self {
            Self::Dense(weights) => weights,
            Self::Factored(weights) => weights,
        }
    }
}

impl WeightMatrix for Weights {
    fn neurons(&self) -> usize {
        self.inner().neurons()
    }

    fn local_fields_into(&self, state: ArrayView1<'_, i8>, out: ArrayViewMut1<'_, f64>) {
        self.inner().local_fields_into(state, out);
    }

    fn local_field(&self, i: usize, state: ArrayView1<'_, i8>) -> f64 {
        self.inner().local_field(i, state)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.inner().to_dense()
    }
}

impl From<DenseWeights> for Weights {
    fn from(weights: DenseWeights) -> Self {
        Self::Dense(weights)
    }
}

impl From<FactoredWeights> for Weights {
    fn from(weights: FactoredWeights) -> Self {
        Self::Factored(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn delegates_to_the_backend() {
        let dense = DenseWeights::new(array![[0.0, 0.5], [0.5, 0.0]]).unwrap();
        let weights = Weights::from(dense.clone());
        assert_eq!(weights.backend(), Backend::Dense);
        assert_eq!(weights.neurons(), 2);
        assert_eq!(
            weights.local_fields(array![1, -1].view()),
            array![-0.5, 0.5]
        );
        assert_eq!(weights.local_field(1, array![1, -1].view()), 0.5);
        assert_eq!(weights.to_dense(), dense.to_dense());
    }
}
