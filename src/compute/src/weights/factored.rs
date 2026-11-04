//! Weights stored as a product of the patterns and a coefficient matrix.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Zip};

use super::{PARALLEL_MIN_NEURONS, WeightMatrix};
use crate::{Error, PatternSet, Result};

/// Weight matrix `W = A Xi - diag(d)` of `N` neurons built from `p` bipolar
/// basis vectors.
///
/// `Xi` (`p x N`) holds the basis vectors (the stored patterns) as rows and
/// `A` (`N x p`) the coefficients; `d_i = sum_mu A[i][mu] Xi[mu][i]` removes
/// the diagonal of `A Xi`, so `w_ii = 0`. The memory is `O(N p)` instead of
/// `O(N^2)`: for `N = 60000` and `p = 12` about 6 MB instead of 28.8 GB.
/// The Hebbian rule is `A = Xi^T / N`.
#[derive(Debug, Clone, PartialEq)]
pub struct FactoredWeights {
    basis: Array2<i8>,
    coefficients: Array2<f64>,
    diagonal: Array1<f64>,
}

impl FactoredWeights {
    /// Builds the weights from the patterns of `basis` (the rows of `Xi`) and
    /// the `N x p` coefficient matrix `A` of finite values.
    pub fn new(basis: &PatternSet, coefficients: Array2<f64>) -> Result<Self> {
        let expected = (basis.neurons(), basis.len());
        if coefficients.dim() != expected {
            return Err(Error::InvalidArgument {
                message: format!(
                    "coefficient matrix must be {}x{}, found {}x{}",
                    expected.0,
                    expected.1,
                    coefficients.nrows(),
                    coefficients.ncols()
                ),
            });
        }
        if let Some(((i, mu), a)) = coefficients.indexed_iter().find(|(_, a)| !a.is_finite()) {
            return Err(Error::InvalidArgument {
                message: format!("coefficients must be finite, found A[{i}][{mu}] = {a}"),
            });
        }
        let mut weights = Self {
            basis: basis.patterns().to_owned(),
            coefficients,
            diagonal: Array1::zeros(expected.0),
        };
        weights.update_diagonal();
        Ok(weights)
    }

    /// Basis vectors `Xi` (`p x N`).
    pub fn basis(&self) -> ArrayView2<'_, i8> {
        self.basis.view()
    }

    /// Coefficients `A` (`N x p`).
    pub fn coefficients(&self) -> ArrayView2<'_, f64> {
        self.coefficients.view()
    }

    /// Diagonal `d` of `A Xi`, subtracted to give `w_ii = 0`.
    pub fn diagonal(&self) -> ArrayView1<'_, f64> {
        self.diagonal.view()
    }

    /// Recomputes `d_i = sum_mu A[i][mu] Xi[mu][i]` from the coefficients.
    fn update_diagonal(&mut self) {
        for (i, (d, a)) in self
            .diagonal
            .iter_mut()
            .zip(self.coefficients.rows())
            .enumerate()
        {
            *d = a
                .iter()
                .zip(self.basis.column(i))
                .map(|(&a, &xi)| a * f64::from(xi))
                .sum();
        }
    }

    /// Overlaps `m = Xi x` of `state` with the basis vectors, computed
    /// exactly in integers. The values are integers of magnitude at most
    /// `N`, so `f64` holds them exactly.
    pub fn overlaps(&self, state: ArrayView1<'_, i8>) -> Array1<f64> {
        assert_eq!(state.len(), self.neurons(), "state length");
        let mut overlaps = Array1::zeros(self.basis.nrows());
        let rows = Zip::from(&mut overlaps).and(self.basis.rows());
        let overlap = |m: &mut f64, row: ArrayView1<'_, i8>| {
            let sum: i64 = row
                .iter()
                .zip(state)
                .map(|(&xi, &x)| i64::from(xi) * i64::from(x))
                .sum();
            *m = sum as f64;
        };
        if self.neurons() >= PARALLEL_MIN_NEURONS {
            rows.par_for_each(overlap);
        } else {
            rows.for_each(overlap);
        }
        overlaps
    }

    /// Local field `h_i = A[i] m - d_i x_i` of neuron `i` in state `x_i`,
    /// given the overlaps `m = Xi x` of the whole state.
    pub(super) fn field_from_overlaps(&self, i: usize, overlaps: &Array1<f64>, x_i: i8) -> f64 {
        self.coefficients.row(i).dot(overlaps) - self.diagonal[i] * f64::from(x_i)
    }
}

impl WeightMatrix for FactoredWeights {
    fn neurons(&self) -> usize {
        self.coefficients.nrows()
    }

    /// Computes `h = A (Xi x) - d * x` in `O(N p)` operations: the overlaps
    /// `m = Xi x` once, then `h_i = A[i] m - d_i x_i` for every neuron.
    fn local_fields_into(&self, state: ArrayView1<'_, i8>, out: ArrayViewMut1<'_, f64>) {
        let overlaps = self.overlaps(state);
        let fields = Zip::indexed(out).and(state);
        let field = |i: usize, h: &mut f64, &x: &i8| {
            *h = self.field_from_overlaps(i, &overlaps, x);
        };
        if self.neurons() >= PARALLEL_MIN_NEURONS {
            fields.par_for_each(field);
        } else {
            fields.for_each(field);
        }
    }

    /// Computes `h_i = A[i] (Xi x) - d_i x_i`. The overlaps cost `O(N p)`,
    /// so computing every field this way costs `O(N^2 p)`; use
    /// [`local_fields_into`](WeightMatrix::local_fields_into) for all
    /// fields at once or a [`FieldTracker`](super::FieldTracker) for
    /// updates of one neuron at a time.
    fn local_field(&self, i: usize, state: ArrayView1<'_, i8>) -> f64 {
        let overlaps = self.overlaps(state);
        self.field_from_overlaps(i, &overlaps, state[i])
    }

    fn to_dense(&self) -> Array2<f64> {
        let basis = self.basis.mapv(f64::from);
        let mut matrix = self.coefficients.dot(&basis);
        matrix.diag_mut().fill(0.0);
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::Pattern;
    use crate::weights::DenseWeights;
    use ndarray::array;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn random(n: usize, p: usize, seed: u64) -> FactoredWeights {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let patterns: Vec<_> = (0..p)
            .map(|_| Pattern::random(n, &mut rng).unwrap())
            .collect();
        let basis = PatternSet::from_patterns(n, 1, &patterns).unwrap();
        let coefficients = Array2::from_shape_fn((n, p), |_| rng.random_range(-1.0..1.0));
        FactoredWeights::new(&basis, coefficients).unwrap()
    }

    #[test]
    fn dense_form_has_zero_diagonal() {
        let weights = random(7, 3, 1);
        let dense = weights.to_dense();
        assert!(dense.diag().iter().all(|&w| w == 0.0));
        let full = weights.coefficients().dot(&weights.basis().mapv(f64::from));
        assert!((dense[[0, 1]] - full[[0, 1]]).abs() < 1e-15);
        assert!((weights.diagonal[2] - full[[2, 2]]).abs() < 1e-12);
    }

    #[test]
    fn local_fields_match_the_dense_matrix() {
        let weights = random(30, 4, 2);
        let dense = DenseWeights::new(weights.to_dense()).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for _ in 0..5 {
            let state = Pattern::random(30, &mut rng).unwrap();
            let expected = dense.local_fields(state.values());
            let fields = weights.local_fields(state.values());
            for i in 0..30 {
                assert!((fields[i] - expected[i]).abs() < 1e-12, "{i}");
                assert!((weights.local_field(i, state.values()) - expected[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn overlaps_are_exact_dot_products() {
        let basis = PatternSet::new(3, 1, array![[1, -1, 1], [1, 1, 1]]).unwrap();
        let weights = FactoredWeights::new(&basis, Array2::zeros((3, 2))).unwrap();
        assert_eq!(weights.overlaps(array![1, 1, -1].view()), array![-1.0, 1.0]);
    }

    #[test]
    fn rejects_a_coefficient_shape_mismatch() {
        let basis = PatternSet::new(3, 1, array![[1, -1, 1]]).unwrap();
        let error = FactoredWeights::new(&basis, Array2::zeros((1, 3))).unwrap_err();
        assert!(error.to_string().contains("must be 3x1"), "{error}");
        let error =
            FactoredWeights::new(&basis, array![[0.0], [f64::INFINITY], [0.0]]).unwrap_err();
        assert!(error.to_string().contains("A[1][0] = inf"), "{error}");
    }
}
