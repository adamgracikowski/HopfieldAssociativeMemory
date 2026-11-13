//! Hebbian weights of a set too large to hold as a pattern set.

use ndarray::Array2;

use super::{Hyperplane, StrictStability};
use crate::weights::{DenseWeights, Weights};

/// The Hebbian weights `W = (1/N) sum_mu xi^mu xi^mu^T` with zero diagonal
/// of all members of `S_u`, accumulated member by member: the sums
/// `C_ij = sum_mu xi_i^mu xi_j^mu` are integers, kept exactly in `i64`, so
/// the weights equal those of [`Hebb`](crate::learning::Hebb) trained on
/// the whole set, without its `p x N` matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HebbSum {
    sums: Array2<i64>,
    patterns: u64,
}

impl HebbSum {
    /// The sums of every member of `plane`, accumulated in parallel chunks
    /// and added exactly.
    pub fn of(plane: &Hyperplane) -> Self {
        let n = plane.neurons();
        let partials = plane.fold_members(
            || (Array2::<i64>::zeros((n, n)), 0u64),
            |(sums, patterns), state| {
                for i in 0..n {
                    for j in 0..n {
                        sums[[i, j]] += i64::from(state[i] * state[j]);
                    }
                }
                *patterns += 1;
            },
        );
        let mut sums = Array2::zeros((n, n));
        let mut patterns = 0;
        for (partial, count) in partials {
            sums += &partial;
            patterns += count;
        }
        Self { sums, patterns }
    }

    /// Number of summed patterns, `|S_u|`.
    pub fn patterns(&self) -> u64 {
        self.patterns
    }

    /// The dense weights `C / N` with the diagonal set to zero.
    pub fn weights(&self) -> Weights {
        let n = self.sums.nrows() as f64;
        let mut matrix = self.sums.mapv(|c| c as f64 / n);
        matrix.diag_mut().fill(0.0);
        DenseWeights::new(matrix)
            .expect("the sums give a square matrix")
            .into()
    }

    /// Strict stability of every member of `plane` in the network of
    /// [`weights`](Self::weights), computed exactly: `N h_i = sum_{j != i}
    /// C_ij xi_j` is an integer, so a member is strictly stable when every
    /// `xi_i N h_i > 0`, without the rounding of large `f64` fields (the
    /// sums grow with `|S_u|`). The reported aligned fields are `xi_i h_i`
    /// in the units of the weights. Chunks are combined in chunk order.
    ///
    /// # Panics
    ///
    /// Panics if `plane` has a different number of neurons or no members.
    pub fn strict_stability(&self, plane: &Hyperplane) -> StrictStability {
        let n = plane.neurons();
        assert_eq!(self.sums.nrows(), n, "sums and hyperplane size");
        let partials = plane.fold_members(
            || (0u64, 0u64, i64::MAX, 0i128),
            |(patterns, stable, min, sum), state| {
                let mut all = true;
                for i in 0..n {
                    let field: i64 = (0..n)
                        .filter(|&j| j != i)
                        .map(|j| self.sums[[i, j]] * i64::from(state[j]))
                        .sum();
                    let aligned = i64::from(state[i]) * field;
                    all &= aligned > 0;
                    *min = (*min).min(aligned);
                    *sum += i128::from(aligned);
                }
                *patterns += 1;
                *stable += u64::from(all);
            },
        );
        let (mut patterns, mut stable, mut min, mut sum) = (0u64, 0u64, i64::MAX, 0i128);
        for (p, s, m, total) in partials {
            patterns += p;
            stable += s;
            min = min.min(m);
            sum += total;
        }
        assert!(patterns > 0, "the hyperplane has members");
        let scale = n as f64;
        StrictStability {
            patterns,
            stable,
            min_aligned: min as f64 / scale,
            mean_aligned: sum as f64 / (patterns as f64 * scale * scale),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PatternSet;
    use crate::learning::{Hebb, LearningRule};
    use crate::weights::{Backend, WeightMatrix};

    /// The exact check agrees with the check of the `f64` weights on a set
    /// with small fields: the same counts, the same smallest and mean
    /// aligned field up to rounding.
    #[test]
    fn exact_stability_agrees_with_the_floating_point_check() {
        for weights in [
            vec![1, 1, 1, 1, 1, 1, 1, 1, 2],
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        ] {
            let plane = Hyperplane::new(weights).unwrap();
            let sum = HebbSum::of(&plane);
            let exact = sum.strict_stability(&plane);
            let float = StrictStability::of(&sum.weights(), &plane);
            assert_eq!(
                (exact.patterns, exact.stable),
                (float.patterns, float.stable)
            );
            assert!((exact.min_aligned - float.min_aligned).abs() < 1e-9);
            assert!((exact.mean_aligned - float.mean_aligned).abs() < 1e-9);
        }
    }

    /// The accumulated weights equal those of the Hebbian rule trained on
    /// the members as a pattern set.
    #[test]
    fn equals_the_hebbian_rule_on_the_whole_set() {
        let plane = Hyperplane::new(vec![1, 2, 1, 1, 3, 2, 1, 1]).unwrap();
        let members: Vec<_> = plane.members().collect();
        let set = PatternSet::from_patterns(plane.neurons(), 1, &members).unwrap();
        let trained = Hebb.train(&set, Backend::Dense).unwrap().weights.to_dense();
        let sum = HebbSum::of(&plane);
        assert_eq!(sum.patterns(), members.len() as u64);
        let accumulated = sum.weights().to_dense();
        for (a, b) in accumulated.iter().zip(&trained) {
            assert!((a - b).abs() < 1e-12, "{a} {b}");
        }
    }
}
