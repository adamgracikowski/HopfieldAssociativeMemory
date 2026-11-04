//! Local fields of a state that changes one neuron at a time.

use ndarray::Array1;

use super::{DenseWeights, FactoredWeights, Weights};
use crate::patterns::{BipolarVector, Pattern, PatternView};

/// A network state together with what its local fields need, so that one
/// field costs `O(N)` for dense weights and `O(p)` for factored weights,
/// instead of recomputing the fields of the whole state.
///
/// - Dense weights: the state is also kept in `f64`, and `h_i` is the dot
///   product of row `i` with it (`O(N)`).
/// - Factored weights `W = A Xi - diag(d)`: the overlaps `m = Xi x` are
///   kept, `h_i = A[i] m - d_i x_i` costs `O(p)`, and flipping neuron `j`
///   updates them by `m += 2 x_j Xi[:, j]` (with the new `x_j`), also
///   `O(p)`. The overlaps are integers of magnitude at most `N`, so they
///   stay exact in `f64`.
///
/// Every field is bitwise equal to the corresponding element of
/// [`WeightMatrix::local_fields`](super::WeightMatrix::local_fields) of the
/// current state.
#[derive(Debug, Clone)]
pub struct FieldTracker<'w> {
    state: Array1<i8>,
    cache: Cache<'w>,
}

#[derive(Debug, Clone)]
enum Cache<'w> {
    Dense {
        weights: &'w DenseWeights,
        state: Array1<f64>,
    },
    Factored {
        weights: &'w FactoredWeights,
        overlaps: Array1<f64>,
    },
}

impl<'w> FieldTracker<'w> {
    /// Tracks the fields of `weights` starting from `state`.
    ///
    /// # Panics
    ///
    /// Panics if `state` does not have `N` elements.
    pub fn new(weights: &'w Weights, state: PatternView<'_>) -> Self {
        let values = state.values();
        let cache = match weights {
            Weights::Dense(weights) => {
                assert_eq!(values.len(), weights.matrix().nrows(), "state length");
                Cache::Dense {
                    weights,
                    state: values.mapv(f64::from),
                }
            }
            Weights::Factored(weights) => Cache::Factored {
                weights,
                overlaps: weights.overlaps(values),
            },
        };
        Self {
            state: values.to_owned(),
            cache,
        }
    }

    /// Number of neurons `N`.
    pub fn neurons(&self) -> usize {
        self.state.len()
    }

    /// The current state.
    pub fn state(&self) -> PatternView<'_> {
        BipolarVector::new_unchecked(self.state.view())
    }

    /// Local field `h_i` of the current state.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    pub fn field(&self, i: usize) -> f64 {
        match &self.cache {
            Cache::Dense { weights, state } => weights.matrix().row(i).dot(state),
            Cache::Factored { weights, overlaps } => {
                weights.field_from_overlaps(i, overlaps, self.state[i])
            }
        }
    }

    /// Flips neuron `i` and updates the cached values.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    pub fn flip(&mut self, i: usize) {
        let x = -self.state[i];
        self.state[i] = x;
        match &mut self.cache {
            Cache::Dense { state, .. } => state[i] = f64::from(x),
            Cache::Factored { weights, overlaps } => {
                let change = 2.0 * f64::from(x);
                for (m, &xi) in overlaps.iter_mut().zip(weights.basis().column(i)) {
                    *m += change * f64::from(xi);
                }
            }
        }
    }

    /// Consumes the tracker and returns the current state.
    pub fn into_state(self) -> Pattern {
        BipolarVector::new_unchecked(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PatternSet;
    use crate::learning::{Hebb, LearningRule};
    use crate::weights::{Backend, WeightMatrix};
    use ndarray::Array2;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn random_set(n: usize, p: usize, rng: &mut ChaCha8Rng) -> PatternSet {
        let patterns: Vec<_> = (0..p).map(|_| Pattern::random(n, rng).unwrap()).collect();
        PatternSet::from_patterns(n, 1, &patterns).unwrap()
    }

    /// Hebbian weights of both backends and asymmetric factored weights
    /// with random coefficients.
    fn networks(rng: &mut ChaCha8Rng) -> Vec<Weights> {
        let set = random_set(40, 4, rng);
        let coefficients = Array2::from_shape_fn((40, 4), |_| rng.random_range(-1.0..1.0));
        vec![
            Hebb.train(&set, Backend::Dense).unwrap().weights,
            Hebb.train(&set, Backend::Factored).unwrap().weights,
            FactoredWeights::new(&set, coefficients).unwrap().into(),
        ]
    }

    #[test]
    fn fields_follow_the_state_through_flips() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for weights in networks(&mut rng) {
            let start = Pattern::random(40, &mut rng).unwrap();
            let mut tracker = FieldTracker::new(&weights, start.view());
            for _ in 0..100 {
                tracker.flip(rng.random_range(0..40));
                let expected = weights.local_fields(tracker.state().values());
                for i in 0..40 {
                    assert_eq!(tracker.field(i), expected[i], "{:?} {i}", weights.backend());
                }
            }
            if let Cache::Factored { weights, overlaps } = &tracker.cache {
                assert_eq!(overlaps, weights.overlaps(tracker.state().values()));
            }
        }
    }

    #[test]
    fn flipping_twice_restores_the_state() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let weights = networks(&mut rng).remove(1);
        let start = Pattern::random(40, &mut rng).unwrap();
        let mut tracker = FieldTracker::new(&weights, start.view());
        tracker.flip(3);
        assert_eq!(start.hamming(&tracker.state()), 1);
        tracker.flip(3);
        assert_eq!(tracker.neurons(), 40);
        assert_eq!(tracker.into_state(), start);
    }
}
