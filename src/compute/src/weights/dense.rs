//! Weights stored as a full `N x N` matrix.

use ndarray::linalg::general_mat_vec_mul;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

use super::WeightMatrix;
use crate::{Error, Result};

/// Full `N x N` weight matrix with zero diagonal. Symmetry is not required:
/// some learning rules produce asymmetric weights.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseWeights {
    matrix: Array2<f64>,
}

impl DenseWeights {
    /// Wraps `matrix` after checking that it is a non-empty square matrix
    /// with zero diagonal.
    pub fn new(matrix: Array2<f64>) -> Result<Self> {
        let (rows, cols) = matrix.dim();
        if rows == 0 || rows != cols {
            return Err(invalid(format!(
                "weight matrix must be square and non-empty, found {rows}x{cols}"
            )));
        }
        if let Some((i, w)) = matrix.diag().indexed_iter().find(|(_, w)| **w != 0.0) {
            return Err(invalid(format!(
                "weight matrix must have zero diagonal, found w[{i}][{i}] = {w}"
            )));
        }
        Ok(Self { matrix })
    }

    /// Zero matrix of `n` neurons.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn zeros(n: usize) -> Self {
        assert!(n > 0, "a network has at least one neuron");
        Self {
            matrix: Array2::zeros((n, n)),
        }
    }

    /// The matrix.
    pub fn matrix(&self) -> ArrayView2<'_, f64> {
        self.matrix.view()
    }

    /// Consumes the weights and returns the matrix.
    pub fn into_inner(self) -> Array2<f64> {
        self.matrix
    }
}

impl WeightMatrix for DenseWeights {
    fn neurons(&self) -> usize {
        self.matrix.nrows()
    }

    /// Computes `h = W x` with a matrix-vector product, converting the state
    /// to `f64` once.
    fn local_fields_into(&self, state: ArrayView1<'_, i8>, mut out: ArrayViewMut1<'_, f64>) {
        let x: Array1<f64> = state.mapv(f64::from);
        general_mat_vec_mul(1.0, &self.matrix, &x, 0.0, &mut out);
    }

    fn local_field(&self, i: usize, state: ArrayView1<'_, i8>) -> f64 {
        assert_eq!(state.len(), self.neurons(), "state length");
        self.matrix
            .row(i)
            .iter()
            .zip(state)
            .map(|(&w, &x)| w * f64::from(x))
            .sum()
    }

    fn to_dense(&self) -> Array2<f64> {
        self.matrix.clone()
    }
}

fn invalid(message: String) -> Error {
    Error::InvalidArgument { message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample() -> DenseWeights {
        DenseWeights::new(array![[0.0, 1.0, -2.0], [1.0, 0.0, 0.5], [-2.0, 0.5, 0.0]]).unwrap()
    }

    #[test]
    fn computes_local_fields() {
        let weights = sample();
        let state = array![1, -1, 1];
        let fields = weights.local_fields(state.view());
        assert_eq!(fields, array![-3.0, 1.5, -2.5]);
        for i in 0..3 {
            assert_eq!(weights.local_field(i, state.view()), fields[i]);
        }
    }

    #[test]
    fn rejects_invalid_matrices() {
        assert!(DenseWeights::new(Array2::zeros((2, 3))).is_err());
        assert!(DenseWeights::new(Array2::zeros((0, 0))).is_err());
        let error = DenseWeights::new(array![[0.0, 1.0], [1.0, 0.1]]).unwrap_err();
        assert!(error.to_string().contains("zero diagonal"), "{error}");
    }

    #[test]
    fn zeros_has_no_fields() {
        let weights = DenseWeights::zeros(4);
        assert_eq!(weights.neurons(), 4);
        assert_eq!(
            weights.local_fields(array![1, 1, -1, 1].view()),
            Array1::zeros(4)
        );
        assert_eq!(sample().into_inner(), sample().to_dense());
    }
}
