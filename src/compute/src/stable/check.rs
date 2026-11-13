//! Strict stability of every member of a hyperplane set.

use ndarray::{Array1, ArrayView1};
use serde::Serialize;

use super::Hyperplane;
use crate::dynamics::ZERO_FIELD_TOLERANCE;
use crate::weights::WeightMatrix;

/// How many members of `S_u` are strict fixed points of a network: every
/// pixel has the aligned field `xi_i h_i > 1e-10`
/// ([`ZERO_FIELD_TOLERANCE`]), so the state stays whatever `sgn(0)` means.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct StrictStability {
    /// Members checked, `|S_u|`.
    pub patterns: u64,
    /// Members whose every aligned field exceeds the tolerance.
    pub stable: u64,
    /// Smallest aligned field `xi_i h_i` over all members and pixels.
    pub min_aligned: f64,
    /// Mean aligned field over all members and pixels.
    pub mean_aligned: f64,
}

/// Partial result of one chunk of members.
struct Partial {
    fields: Array1<f64>,
    patterns: u64,
    stable: u64,
    min: f64,
    sum: f64,
}

impl StrictStability {
    /// Checks every member of `plane` against `weights`, computing the
    /// local fields of each member one at a time. The chunks of members
    /// are checked in parallel and combined in chunk order, so the result
    /// does not depend on the number of threads.
    ///
    /// # Panics
    ///
    /// Panics if `weights` and `plane` have different numbers of neurons or
    /// `S_u` is empty.
    pub fn of(weights: &(impl WeightMatrix + Sync), plane: &Hyperplane) -> Self {
        let n = plane.neurons();
        assert_eq!(weights.neurons(), n, "weights and hyperplane size");
        let partials = plane.fold_members(
            || Partial {
                fields: Array1::zeros(n),
                patterns: 0,
                stable: 0,
                min: f64::INFINITY,
                sum: 0.0,
            },
            |partial, state| {
                let state = ArrayView1::from(state);
                weights.local_fields_into(state, partial.fields.view_mut());
                let mut stable = true;
                for (&x, &h) in state.iter().zip(&partial.fields) {
                    let aligned = f64::from(x) * h;
                    stable &= aligned > ZERO_FIELD_TOLERANCE;
                    partial.min = partial.min.min(aligned);
                    partial.sum += aligned;
                }
                partial.patterns += 1;
                partial.stable += u64::from(stable);
            },
        );
        let mut total = Partial {
            fields: Array1::zeros(0),
            patterns: 0,
            stable: 0,
            min: f64::INFINITY,
            sum: 0.0,
        };
        for partial in partials {
            total.patterns += partial.patterns;
            total.stable += partial.stable;
            total.min = total.min.min(partial.min);
            total.sum += partial.sum;
        }
        assert!(total.patterns > 0, "the hyperplane has members");
        Self {
            patterns: total.patterns,
            stable: total.stable,
            min_aligned: total.min,
            mean_aligned: total.sum / (total.patterns as f64 * n as f64),
        }
    }

    /// Whether every member is a strict fixed point.
    pub fn all_stable(&self) -> bool {
        self.stable == self.patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PatternSet;
    use crate::dynamics::{Stability, ZeroField};
    use crate::learning::{Hebb, LearningRule};
    use crate::weights::Backend;

    /// The members of a small hyperplane trained with the Hebbian rule: the
    /// streaming check counts the same strictly stable patterns as
    /// [`Stability`] with aligned fields required to be positive.
    #[test]
    fn agrees_with_the_aligned_fields_of_every_pattern() {
        for weights in [vec![1, 1, 1, 1, 1, 1, 1, 1, 2], vec![1, 2, 1, 2, 3, 1, 2]] {
            let plane = Hyperplane::new(weights).unwrap();
            let members: Vec<_> = plane.members().collect();
            let set = PatternSet::from_patterns(plane.neurons(), 1, &members).unwrap();
            let trained = Hebb.train(&set, Backend::Dense).unwrap();
            let check = StrictStability::of(&trained.weights, &plane);
            let aligned = Stability::aligned_fields(&trained.weights, &set);
            let stable = aligned
                .rows()
                .into_iter()
                .filter(|row| row.iter().all(|&a| a > ZERO_FIELD_TOLERANCE))
                .count() as u64;
            assert_eq!(check.patterns, members.len() as u64);
            assert_eq!(check.stable, stable);
            let min = aligned.iter().copied().fold(f64::INFINITY, f64::min);
            assert_eq!(check.min_aligned, min);
            let mean = aligned.sum() / aligned.len() as f64;
            assert!((check.mean_aligned - mean).abs() < 1e-12);
            let keep = Stability::of(&trained.weights, &set, ZeroField::Keep);
            assert!(check.stable as usize <= keep.stable);
        }
    }
}
