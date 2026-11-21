//! Stability of the stored patterns.

use ndarray::{Array2, ArrayView2};
use serde::Serialize;

use super::ZeroField;
use crate::PatternSet;
use crate::weights::{WeightMatrix, Weights};

/// Which stored patterns are fixed points of the network (`sgn(W xi) = xi`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Stability {
    /// Number of stable patterns.
    pub stable: usize,
    /// For every pattern, the number of neurons that would change.
    pub unstable_neurons: Vec<usize>,
}

impl Stability {
    /// Checks every pattern of `set`, with `zero_field` deciding zero fields.
    pub fn of(weights: &Weights, set: &PatternSet, zero_field: ZeroField) -> Self {
        Self::from_aligned_fields(Self::aligned_fields(weights, set).view(), set, zero_field)
    }

    /// The aligned local fields `xi_i^mu h_i`, `h = W xi^mu`, as a `p x N`
    /// matrix; a negative one marks a neuron that would change.
    pub fn aligned_fields(weights: &Weights, set: &PatternSet) -> Array2<f64> {
        let mut aligned = Array2::zeros((set.len(), set.neurons()));
        for (pattern, mut row) in set.iter().zip(aligned.rows_mut()) {
            weights.local_fields_into(pattern.values(), row.view_mut());
            row.zip_mut_with(&pattern.values(), |h, &x| *h *= f64::from(x));
        }
        aligned
    }

    /// Stability from the [aligned fields](Self::aligned_fields) of `set`.
    ///
    /// # Panics
    ///
    /// Panics if `aligned` is not `p x N`.
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

    /// Orthogonal patterns with Hebbian weights have `h = xi / 2`.
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

    /// Orthogonal patterns: every aligned field is the signal
    /// `1 - p/N = 1/2`, with no crosstalk.
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

    /// Patterns `(1, 1, 1)` and `(1, 1, -1)`: `w_01 = 2/3`, `w_02 = w_12 = 0`,
    /// so both have aligned fields `(2/3, 2/3, 0)`. Neuron 2 has a zero
    /// field: stable with `keep`, while `plus-one` changes the `-1` of
    /// pattern 1.
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
