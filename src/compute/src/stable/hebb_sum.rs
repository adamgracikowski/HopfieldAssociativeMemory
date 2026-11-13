//! Hebbian weights of a set too large to hold as a pattern set.

use ndarray::Array2;

use super::Hyperplane;
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
            .expect("the sums are symmetric")
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PatternSet;
    use crate::learning::{Hebb, LearningRule};
    use crate::weights::{Backend, WeightMatrix};

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
