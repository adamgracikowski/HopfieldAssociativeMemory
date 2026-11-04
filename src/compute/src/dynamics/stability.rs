//! Stability of the stored patterns.

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
        let unstable_neurons: Vec<usize> = set
            .iter()
            .map(|pattern| {
                let fields = weights.local_fields(pattern.values());
                fields
                    .iter()
                    .zip(pattern.values())
                    .filter(|&(&h, &x)| zero_field.activate(h, x) != x)
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
    use crate::learning::{Hebb, LearningRule};
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
