//! Hebbian learning rule.

use ndarray::Array2;
use ndarray::linalg::general_mat_mul;

use super::{LearningRule, Trained};
use crate::weights::{Backend, DenseWeights, FactoredWeights, Weights};
use crate::{PatternSet, Result};

/// Hebbian rule: `w_ij = (1/N) sum_mu xi_i^mu xi_j^mu` for `i != j` and
/// `w_ii = 0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Hebb;

impl LearningRule for Hebb {
    /// Dense: `W = Xi^T Xi / N` with the diagonal zeroed; factored:
    /// `A = Xi^T / N`.
    fn train(&self, set: &PatternSet, backend: Backend) -> Result<Trained> {
        let n = set.neurons();
        let scale = 1.0 / n as f64;
        let patterns = set.patterns().mapv(f64::from);
        let weights: Weights = match backend.resolve(n) {
            Backend::Factored => {
                let coefficients = patterns.t().mapv(|xi| xi * scale);
                FactoredWeights::new(set, coefficients)?.into()
            }
            _ => {
                let mut matrix = Array2::zeros((n, n));
                general_mat_mul(scale, &patterns.t(), &patterns, 0.0, &mut matrix);
                matrix.diag_mut().fill(0.0);
                DenseWeights::new(matrix)?.into()
            }
        };
        Ok(Trained {
            weights,
            epochs: 0,
            converged: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::Pattern;
    use crate::weights::WeightMatrix;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// 5x5 set: `a` all ones, `b` -1 at pixels 0-9, `c` 1 in the first
    /// column.
    fn three_patterns() -> PatternSet {
        let data = Array2::from_shape_fn((3, 25), |(mu, i)| match mu {
            0 => 1,
            1 if i < 10 => -1,
            1 => 1,
            _ if i % 5 == 0 => 1,
            _ => -1,
        });
        PatternSet::new(5, 5, data).unwrap()
    }

    /// `w_ij = (a_i a_j + b_i b_j + c_i c_j) / 25`:
    /// - `w[0][1]`: `1 + (-1)(-1) + (1)(-1) = 1`, so 0.04,
    /// - `w[0][5]`: `1 + 1 + 1 = 3`, so 0.12,
    /// - `w[0][10]`: `1 + (-1)(1) + 1 = 1`, so 0.04,
    /// - `w[0][11]`: `1 + (-1)(1) + (1)(-1) = -1`, so -0.04,
    /// - `w[1][2]`: `1 + 1 + 1 = 3`, so 0.12,
    /// - `w[1][11]`: `1 + (-1)(1) + (-1)(-1) = 1`, so 0.04.
    #[test]
    fn dense_weights_for_n_25_match_hand_computed_values() {
        let w = Hebb
            .train(&three_patterns(), Backend::Dense)
            .unwrap()
            .weights
            .to_dense();
        for (i, j, expected) in [
            (0, 1, 0.04),
            (0, 5, 0.12),
            (0, 10, 0.04),
            (0, 11, -0.04),
            (1, 2, 0.12),
            (1, 11, 0.04),
        ] {
            assert!(
                (w[[i, j]] - expected).abs() < 1e-15,
                "w[{i}][{j}] = {}",
                w[[i, j]]
            );
            assert_eq!(w[[i, j]], w[[j, i]]);
        }
    }

    #[test]
    fn dense_weights_match_the_definition() {
        let set = three_patterns();
        let w = Hebb.train(&set, Backend::Dense).unwrap().weights.to_dense();
        let xi = set.patterns();
        for i in 0..25 {
            for j in 0..25 {
                let sum: i32 = (0..3)
                    .map(|mu| i32::from(xi[[mu, i]]) * i32::from(xi[[mu, j]]))
                    .sum();
                let expected = if i == j { 0.0 } else { f64::from(sum) / 25.0 };
                assert!((w[[i, j]] - expected).abs() < 1e-15, "w[{i}][{j}]");
                assert_eq!(w[[i, j]], w[[j, i]]);
            }
        }
    }

    #[test]
    fn backends_give_the_same_weights_and_fields() {
        let set = three_patterns();
        let dense = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let factored = Hebb.train(&set, Backend::Factored).unwrap().weights;
        assert_eq!(factored.backend(), Backend::Factored);
        let (a, b) = (dense.to_dense(), factored.to_dense());
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-15));

        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for _ in 0..10 {
            let state = Pattern::random(25, &mut rng).unwrap();
            let (h, g) = (
                dense.local_fields(state.values()),
                factored.local_fields(state.values()),
            );
            assert!(h.iter().zip(&g).all(|(x, y)| (x - y).abs() < 1e-12));
        }
    }

    /// Orthogonal patterns have no crosstalk: `h = (1 - p/N) xi = xi / 2`
    /// for `p = 2`, `N = 4` (the removed diagonal takes `p/N`).
    #[test]
    fn orthogonal_patterns_are_fixed_points() {
        let set = PatternSet::new(4, 1, array![[1, 1, 1, 1], [1, -1, 1, -1]]).unwrap();
        for backend in [Backend::Dense, Backend::Factored] {
            let weights = Hebb.train(&set, backend).unwrap().weights;
            for xi in set.iter() {
                let h = weights.local_fields(xi.values());
                assert_eq!(h, xi.values().mapv(|v| f64::from(v) / 2.0));
            }
        }
    }

    #[test]
    fn auto_uses_the_dense_backend_for_small_networks() {
        let trained = Hebb.train(&three_patterns(), Backend::Auto).unwrap();
        assert_eq!(trained.weights.backend(), Backend::Dense);
        assert_eq!((trained.epochs, trained.converged), (0, true));
    }
}
