//! Oja's single-neuron rule applied to every row of the weight matrix.

use ndarray::{Array1, Array2, Axis};

/// One batch epoch of the single-neuron rule for weights `W = A Xi`.
///
/// With `x = xi / sqrt(N)` and `G = Xi Xi^T` (`gram`), the outputs are
/// `y_mu = A G[:, mu] / sqrt(N)`. The averaged update
/// `dW = (eta / p) sum_mu (y_mu x_mu^T - diag(y_mu * y_mu) W)` keeps the
/// form `dA Xi` with
///
/// `dA = (eta / p) (A G / N - D A)`, `D_ii = sum_mu (A G)[i][mu]^2 / N`,
///
/// because `D` scales whole rows. Every epoch costs `O(N p^2)`. Returns
/// `dA`; `rate` is `eta / p`.
pub(super) fn step(coefficients: &Array2<f64>, gram: &Array2<f64>, rate: f64) -> Array2<f64> {
    let n = coefficients.nrows() as f64;
    let outputs = coefficients.dot(gram);
    let decay: Array1<f64> = outputs.map_axis(Axis(1), |row| row.dot(&row) / n);
    (outputs / n - coefficients * &decay.insert_axis(Axis(1))) * rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// For a single pattern `xi` of `N = 2` neurons, `G = [N]`, so
    /// `dA = rate (N a / N - N a^2 a) = rate a (1 - N a^2)` per row: rows
    /// with `a = 1/2` (Hebb, `xi / N`) grow to `1 / sqrt(2)`, the unit
    /// norm of the row `a xi`.
    #[test]
    fn rows_grow_towards_unit_norm() {
        let gram = array![[2.0]];
        let a = array![[0.5], [-0.5]];
        let delta = step(&a, &gram, 0.1);
        assert!((delta[[0, 0]] - 0.025).abs() < 1e-15, "{delta}");
        assert!((delta[[1, 0]] + 0.025).abs() < 1e-15, "{delta}");

        let unit = array![[0.5f64.sqrt()], [-(0.5f64.sqrt())]];
        assert!(step(&unit, &gram, 0.1).iter().all(|d| d.abs() < 1e-15));
    }
}
