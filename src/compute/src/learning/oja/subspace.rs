//! Oja's subspace rule computed on `p x p` matrices.

use ndarray::Array2;

/// One batch epoch of the subspace rule for `W = Xi^T M Xi / N`, with
/// `G = Xi Xi^T / N` (`gram`) and `rate = eta / p`:
/// `dM = rate M G (I - G M G M)`; `O(p^3)` per epoch.
pub(super) fn step(matrix: &Array2<f64>, gram: &Array2<f64>, rate: f64) -> Array2<f64> {
    let p = gram.nrows();
    let mg = matrix.dot(gram);
    let gmgm = gram.dot(&mg).dot(matrix);
    mg.dot(&(Array2::eye(p) - gmgm)) * rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn inverse_gram_matrix_is_a_fixed_point() {
        let gram = array![[1.0, 0.2], [0.2, 1.0]];
        let inverse = array![[1.0, -0.2], [-0.2, 1.0]] / 0.96;
        let delta = step(&inverse, &gram, 0.05);
        assert!(delta.iter().all(|d| d.abs() < 1e-15), "{delta}");
    }

    #[test]
    fn keeps_a_symmetric_matrix_symmetric() {
        let gram = array![[1.0, 0.3, -0.1], [0.3, 1.0, 0.2], [-0.1, 0.2, 1.0]];
        let mut matrix: Array2<f64> = Array2::eye(3);
        for _ in 0..10 {
            matrix += &step(&matrix, &gram, 0.05);
        }
        assert!(
            matrix
                .iter()
                .zip(matrix.t())
                .all(|(a, b)| (a - b).abs() < 1e-15)
        );
    }
}
