//! Stability of the stored patterns.

use ndarray::{Array2, ArrayView2};
use serde::Serialize;

use super::ZeroField;
use crate::PatternSet;
use crate::weights::{WeightMatrix, Weights};

/// Which stored patterns are fixed points of the network: pattern `mu` is
/// stable when no neuron would change, `sgn(h_i) = xi_i` for the local
/// fields `h = W xi`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Stability {
    /// Number of stable patterns.
    pub stable: usize,
    /// For every pattern, the number of neurons that would change.
    pub unstable_neurons: Vec<usize>,
}

impl Stability {
    /// Checks every pattern of `set` against `weights`, with `zero_field`
    /// deciding neurons with a zero local field. Costs one evaluation of
    /// all local fields per pattern.
    pub fn of(weights: &Weights, set: &PatternSet, zero_field: ZeroField) -> Self {
        Self::from_aligned_fields(Self::aligned_fields(weights, set).view(), set, zero_field)
    }

    /// The aligned local fields `xi_i^mu h_i` of every pattern `mu` of `set`
    /// as a `p x N` matrix, where `h = W xi^mu`. Neuron `i` of pattern `mu`
    /// keeps its value when the aligned field is positive and changes when
    /// it is negative; a zero field is decided by [`ZeroField`]. For the
    /// Hebbian rule the aligned field is the signal `1 - p/N` plus the
    /// crosstalk of the other patterns.
    pub fn aligned_fields(weights: &Weights, set: &PatternSet) -> Array2<f64> {
        let mut aligned = Array2::zeros((set.len(), set.neurons()));
        for (pattern, mut row) in set.iter().zip(aligned.rows_mut()) {
            weights.local_fields_into(pattern.values(), row.view_mut());
            row.zip_mut_with(&pattern.values(), |h, &x| *h *= f64::from(x));
        }
        aligned
    }

    /// Stability from the [aligned fields](Self::aligned_fields) of `set`:
    /// neuron `i` of pattern `mu` would change when
    /// `zero_field.activate(h_i, xi_i) != xi_i` for the local field
    /// `h_i = xi_i * aligned[mu, i]`.
    ///
    /// # Panics
    ///
    /// Panics if `aligned` is not `p x N` for the `p` patterns of `N`
    /// neurons of `set`.
    pub fn from_aligned_fields(
        aligned: ArrayView2<'_, f64>,
        set: &PatternSet,
        zero_field: ZeroField,
    ) -> Self {
        assert_eq!(aligned.dim(), (set.len(), set.neurons()));
        let unstable_neurons: Vec<usize> = set
            .iter()
            .zip(aligned.rows())
            .map(|(pattern, row)| {
                row.iter()
                    .zip(pattern.values())
                    .filter(|&(&a, &x)| zero_field.activate(a * f64::from(x), x) != x)
                    .count()
            })
            .collect();
        Self {
            stable: unstable_neurons.iter().filter(|&&count| count == 0).count(),
            unstable_neurons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::{Hebb, LearningRule, Oja};
    use crate::weights::Backend;
    use ndarray::array;

    /// Orthogonal patterns of `N = 4` neurons: with Hebbian weights every
    /// pattern has `h = xi (N - 2) / N = xi / 2` and is stable.
    #[test]
    fn orthogonal_patterns_are_stable() {
        let set = PatternSet::new(4, 1, array![[1, 1, 1, 1], [1, -1, 1, -1]]).unwrap();
        for backend in [Backend::Dense, Backend::Factored] {
            let weights = Hebb.train(&set, backend).unwrap().weights;
            let stability = Stability::of(&weights, &set, ZeroField::Keep);
            assert_eq!(stability.stable, 2);
            assert_eq!(stability.unstable_neurons, [0, 0]);
        }
    }

    /// The same orthogonal patterns: `h_i = xi_i / 2`, so every aligned
    /// field `xi_i h_i = xi_i^2 / 2 = 1/2`, the signal `1 - p/N = 1 - 2/4`
    /// with no crosstalk (`m = 0`).
    #[test]
    fn aligned_fields_of_orthogonal_patterns_are_the_signal() {
        let set = PatternSet::new(4, 1, array![[1, 1, 1, 1], [1, -1, 1, -1]]).unwrap();
        for backend in [Backend::Dense, Backend::Factored] {
            let weights = Hebb.train(&set, backend).unwrap().weights;
            let aligned = Stability::aligned_fields(&weights, &set);
            assert_eq!(aligned.dim(), (2, 4));
            for &a in &aligned {
                assert!((a - 0.5).abs() < 1e-12, "{aligned}");
            }
        }
    }

    /// Patterns `(1, 1, 1)` and `(1, 1, -1)` of `N = 3`: Hebbian weights
    /// `w_01 = 2/3`, `w_02 = w_12 = 0`. Pattern 0 has `h = (2/3, 2/3, 0)`,
    /// aligned `(2/3, 2/3, 0)`; pattern 1 the same fields, aligned
    /// `(2/3, 2/3, 0)` too (`-1 * 0`). Neuron 2 has a zero field: stable with
    /// `keep`; with `plus-one` the `-1` of pattern 1 would change.
    #[test]
    fn stability_follows_the_aligned_fields() {
        let set = PatternSet::new(3, 1, array![[1, 1, 1], [1, 1, -1]]).unwrap();
        let weights = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let aligned = Stability::aligned_fields(&weights, &set);
        let expected = array![[2.0, 2.0, 0.0], [2.0, 2.0, 0.0]] / 3.0;
        for (a, e) in aligned.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-12, "{aligned}");
        }
        let keep = Stability::from_aligned_fields(aligned.view(), &set, ZeroField::Keep);
        assert_eq!((keep.stable, keep.unstable_neurons), (2, vec![0, 0]));
        let plus_one = Stability::from_aligned_fields(aligned.view(), &set, ZeroField::PlusOne);
        assert_eq!(
            (plus_one.stable, plus_one.unstable_neurons),
            (1, vec![0, 1])
        );
    }

    /// On a correlated set the count of negative aligned fields is the
    /// number of unstable neurons, for the Hebbian and Oja weights.
    #[test]
    fn negative_aligned_fields_are_unstable_neurons() {
        let set = PatternSet::new(
            3,
            2,
            array![
                [1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1],
                [-1, 1, 1, 1, 1, -1],
                [1, -1, 1, -1, -1, -1]
            ],
        )
        .unwrap();
        for weights in [
            Hebb.train(&set, Backend::Dense).unwrap().weights,
            Oja::default().train(&set, Backend::Dense).unwrap().weights,
        ] {
            let aligned = Stability::aligned_fields(&weights, &set);
            let stability = Stability::of(&weights, &set, ZeroField::Keep);
            let negative: Vec<usize> = aligned
                .rows()
                .into_iter()
                .map(|row| row.iter().filter(|&&a| a < -1e-10).count())
                .collect();
            assert_eq!(stability.unstable_neurons, negative);
        }
    }

    /// Patterns `(1, 1)` and `(1, -1)` give `w_01 = (1 - 1) / 2 = 0`, so
    /// every field is zero. With `keep` both patterns are stable; with
    /// `plus-one` the `-1` of `(1, -1)` would change.
    #[test]
    fn zero_fields_follow_the_convention() {
        let set = PatternSet::new(2, 1, array![[1, 1], [1, -1]]).unwrap();
        let weights = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let keep = Stability::of(&weights, &set, ZeroField::Keep);
        assert_eq!((keep.stable, keep.unstable_neurons), (2, vec![0, 0]));
        let plus_one = Stability::of(&weights, &set, ZeroField::PlusOne);
        assert_eq!(
            (plus_one.stable, plus_one.unstable_neurons),
            (1, vec![0, 1])
        );
    }
}
