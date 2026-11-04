//! Weight matrices of a Hopfield network.
//!
//! Two storage backends implement [`WeightMatrix`]:
//! - [`DenseWeights`]: the full `N x N` matrix, for small and medium networks,
//! - [`FactoredWeights`]: `W = A Xi - diag(d)` built from the `p` stored
//!   patterns, using `O(N p)` memory, for large bitmaps where the dense
//!   matrix would not fit in memory.
//!
//! [`Weights`] holds either of them and is chosen with [`Backend`].
//! [`FieldTracker`] keeps a state together with what its local fields need,
//! for updating one neuron at a time.

pub mod backend;
pub mod dense;
pub mod factored;
pub mod tracker;

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

pub use backend::Backend;
pub use dense::DenseWeights;
pub use factored::FactoredWeights;
pub use tracker::FieldTracker;

/// Smallest number of neurons for which all local fields are computed in
/// parallel. Every field is computed the same way either way, so the result
/// does not depend on this limit or on the number of threads.
pub const PARALLEL_MIN_NEURONS: usize = 1024;

/// A weight matrix `W` with zero diagonal, read through the local fields
/// `h = W x` of a bipolar state `x`.
pub trait WeightMatrix {
    /// Number of neurons `N`.
    fn neurons(&self) -> usize;

    /// Writes the local fields `h = W x` of `state` into `out`, without
    /// allocating an `N x N` temporary. Networks of at least
    /// [`PARALLEL_MIN_NEURONS`] neurons compute the fields in parallel.
    ///
    /// # Panics
    ///
    /// Panics if `state` or `out` does not have `N` elements.
    fn local_fields_into(&self, state: ArrayView1<'_, i8>, out: ArrayViewMut1<'_, f64>);

    /// Local field `h_i = sum_j w_ij x_j` of neuron `i`, bitwise equal to
    /// element `i` of [`local_fields_into`](Self::local_fields_into).
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

    /// Energy `E = -1/2 x^T W x = -1/2 sum_i x_i h_i` of `state`. For
    /// symmetric weights with zero diagonal it does not increase when one
    /// neuron is updated at a time.
    ///
    /// # Panics
    ///
    /// Panics if `state` does not have `N` elements.
    fn energy(&self, state: ArrayView1<'_, i8>) -> f64 {
        energy_from_fields(state, self.local_fields(state).view())
    }
}

/// Energy `E = -1/2 sum_i x_i h_i` of `state` from its local fields
/// `fields`.
///
/// # Panics
///
/// Panics if `state` and `fields` have different lengths.
pub fn energy_from_fields(state: ArrayView1<'_, i8>, fields: ArrayView1<'_, f64>) -> f64 {
    assert_eq!(state.len(), fields.len(), "state and fields length");
    -0.5 * fields
        .iter()
        .zip(state)
        .map(|(&h, &x)| h * f64::from(x))
        .sum::<f64>()
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
    use crate::PatternSet;
    use crate::learning::{Hebb, LearningRule};
    use crate::patterns::Pattern;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

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

    /// `W = [[0, 0.5], [0.5, 0]]`: `E(x) = -w_01 x_0 x_1`, so `-0.5` for
    /// `(1, 1)` and `(-1, -1)`, and `0.5` for `(1, -1)`.
    #[test]
    fn energy_of_a_two_neuron_network() {
        let weights = DenseWeights::new(array![[0.0, 0.5], [0.5, 0.0]]).unwrap();
        assert_eq!(weights.energy(array![1, 1].view()), -0.5);
        assert_eq!(weights.energy(array![1, -1].view()), 0.5);
        assert_eq!(weights.energy(array![-1, -1].view()), -0.5);
    }

    /// Above [`PARALLEL_MIN_NEURONS`] the fields are computed in parallel;
    /// they must not depend on the number of threads and must match the
    /// single fields.
    #[test]
    fn parallel_fields_do_not_depend_on_the_thread_count() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 2000;
        let patterns: Vec<_> = (0..5)
            .map(|_| Pattern::random(n, &mut rng).unwrap())
            .collect();
        let set = PatternSet::from_patterns(n, 1, &patterns).unwrap();
        let state = Pattern::random(n, &mut rng).unwrap();
        for backend in [Backend::Dense, Backend::Factored] {
            let weights = Hebb.train(&set, backend).unwrap().weights;
            let fields_with = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| weights.local_fields(state.values()))
            };
            let single = fields_with(1);
            assert_eq!(single, fields_with(4), "{backend}");
            for i in [0, 999, 1999] {
                assert_eq!(weights.local_field(i, state.values()), single[i]);
            }
        }
    }
}
