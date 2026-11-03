//! Oja's learning rule applied to a Hopfield network.

pub mod fit;
mod single_neuron;
mod subspace;
pub mod variant;

use ndarray::Array2;
use serde::Deserialize;

pub use fit::OjaFit;
pub use variant::OjaVariant;

use super::LearningRule;
use crate::weights::{Backend, DenseWeights, FactoredWeights, WeightMatrix, Weights};
use crate::{Error, PatternSet, Result};

/// Oja's rule in batch mode: in every epoch the update is averaged over all
/// normalized patterns `x = xi / sqrt(N)`.
///
/// Learning starts from the Hebbian weights divided by
/// `s = max(1, max_mu sum_nu |G[mu][nu]|)`, a Gershgorin bound on the largest
/// eigenvalue of `G = Xi Xi^T / N`, so every eigenvalue of the start is at
/// most 1. Without this, the cubic term of the rule overshoots and diverges
/// for strongly correlated sets, whose Hebbian eigenvalues reach `p`.
///
/// The weights keep the form `W = A Xi` (`A`: `N x p`, `Xi`: `p x N`),
/// because the start and every update are combinations of the patterns, so
/// learning runs on small matrices for both backends (see
/// [`OjaVariant`]): the subspace rule on `M` with `A = Xi^T M / N` at
/// `O(p^3)` per epoch, the single-neuron rule on `A` at `O(N p^2)`.
/// The diagonal is zeroed only after learning, which keeps this form.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Oja {
    /// Form of the rule.
    pub variant: OjaVariant,
    /// Learning rate `eta` for the normalized patterns.
    pub eta: f64,
    /// Largest number of epochs.
    pub max_epochs: usize,
    /// Learning stops when an upper bound on `max |dw_ij|` of an epoch falls
    /// below this value.
    pub tolerance: f64,
}

impl Default for Oja {
    /// Subspace rule, `eta = 0.1`, at most 100000 epochs, tolerance `1e-8`.
    /// The subspace rule learns small eigenvalues of the pattern overlap
    /// matrix slowly (`letters-14x20` needs tens of thousands of epochs), and
    /// its epochs cost only `O(p^3)`.
    fn default() -> Self {
        Self {
            variant: OjaVariant::Subspace,
            eta: 0.1,
            max_epochs: 100_000,
            tolerance: 1e-8,
        }
    }
}

impl Oja {
    /// Runs the rule on `set` and returns the coefficients of `W = A Xi`
    /// with the number of epochs and the convergence flag, or
    /// [`Error::LearningDiverged`] if the weights stop being finite.
    pub fn fit(&self, set: &PatternSet) -> Result<OjaFit> {
        self.validate()?;
        let n = set.neurons() as f64;
        let rate = self.eta / set.len() as f64;
        let patterns = set.patterns().mapv(f64::from);
        let gram = patterns.dot(&patterns.t()) / n;
        let scale = start_scale(&gram);

        match self.variant {
            OjaVariant::Subspace => {
                let mut matrix = Array2::eye(set.len()) / scale;
                let (epochs, converged) = self.iterate(&mut matrix, |matrix| {
                    let delta = subspace::step(matrix, &gram, rate);
                    let bound = delta.iter().map(|d| d.abs()).sum::<f64>() / n;
                    (delta, bound)
                })?;
                Ok(OjaFit {
                    coefficients: patterns.t().dot(&matrix) / n,
                    epochs,
                    converged,
                })
            }
            OjaVariant::SingleNeuron => {
                let gram = gram * n;
                let mut coefficients = patterns.t().mapv(|xi| xi / (n * scale));
                let (epochs, converged) = self.iterate(&mut coefficients, |coefficients| {
                    let delta = single_neuron::step(coefficients, &gram, rate);
                    let bound = delta
                        .rows()
                        .into_iter()
                        .map(|row| row.iter().map(|d| d.abs()).sum::<f64>())
                        .fold(0.0, f64::max);
                    (delta, bound)
                })?;
                Ok(OjaFit {
                    coefficients,
                    epochs,
                    converged,
                })
            }
        }
    }

    /// Adds the update returned by `step` to `state` once per epoch until
    /// the returned bound on `max |dw_ij|` falls below the tolerance or the
    /// epoch limit is reached. Returns the number of epochs and whether the
    /// tolerance was reached.
    fn iterate(
        &self,
        state: &mut Array2<f64>,
        mut step: impl FnMut(&Array2<f64>) -> (Array2<f64>, f64),
    ) -> Result<(usize, bool)> {
        for epoch in 1..=self.max_epochs {
            let (delta, bound) = step(state);
            *state += &delta;
            if !(bound.is_finite() && state.iter().all(|v| v.is_finite())) {
                return Err(Error::LearningDiverged {
                    rule: format!("Oja's {} rule", self.variant),
                    epochs: epoch,
                });
            }
            if bound < self.tolerance {
                return Ok((epoch, true));
            }
        }
        Ok((self.max_epochs, false))
    }

    fn validate(&self) -> Result<()> {
        let invalid = |message: String| Err(Error::InvalidArgument { message });
        if !(self.eta.is_finite() && self.eta > 0.0) {
            return invalid(format!(
                "learning rate must be positive, found {}",
                self.eta
            ));
        }
        if self.max_epochs == 0 {
            return invalid("at least one epoch is needed".to_owned());
        }
        if !(self.tolerance.is_finite() && self.tolerance >= 0.0) {
            return invalid(format!(
                "tolerance must be non-negative, found {}",
                self.tolerance
            ));
        }
        Ok(())
    }
}

impl LearningRule for Oja {
    /// Runs [`fit`](Oja::fit) and stores `W = A Xi` with zero diagonal in
    /// the factored backend, or expands it into the dense one.
    fn train(&self, set: &PatternSet, backend: Backend) -> Result<Weights> {
        let factored = FactoredWeights::new(set, self.fit(set)?.coefficients)?;
        match backend.resolve(set.neurons()) {
            Backend::Factored => Ok(factored.into()),
            _ => Ok(DenseWeights::new(factored.to_dense())?.into()),
        }
    }
}

/// `max(1, max_mu sum_nu |G[mu][nu]|)`, an upper bound on the largest
/// eigenvalue of the Hebbian weights (Gershgorin), used to scale the start.
fn start_scale(gram: &Array2<f64>) -> f64 {
    gram.rows()
        .into_iter()
        .map(|row| row.iter().map(|g| g.abs()).sum::<f64>())
        .fold(1.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;
    use strum::IntoEnumIterator;

    /// Two 5x5 patterns: `a` is all ones, `b` has ones at pixels 0-14 and
    /// -1 at pixels 15-24; their overlap is `a . b = 5`.
    fn two_patterns() -> PatternSet {
        let data = Array2::from_shape_fn((2, 25), |(mu, i)| match mu {
            0 => 1,
            _ if i < 15 => 1,
            _ => -1,
        });
        PatternSet::new(5, 5, data).unwrap()
    }

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

    /// Six 5x5 patterns that differ from all ones in one pixel each, so
    /// the largest eigenvalue of `G` is close to `p = 6`.
    fn correlated_patterns() -> PatternSet {
        let data = Array2::from_shape_fn((6, 25), |(mu, i)| if i == mu { -1 } else { 1 });
        PatternSet::new(5, 5, data).unwrap()
    }

    fn full_matrix(fit: &OjaFit, set: &PatternSet) -> Array2<f64> {
        fit.coefficients.dot(&set.patterns().mapv(f64::from))
    }

    /// Direct `N x N` implementation of the documented batch rules, used as
    /// a reference: start `X^T X / s`, then
    /// `W += (eta / p) (Y^T X - decay)` with `Y = X W^T`.
    fn reference(set: &PatternSet, variant: OjaVariant, eta: f64, epochs: usize) -> Array2<f64> {
        let n = set.neurons() as f64;
        let p = set.len() as f64;
        let x = set.patterns().mapv(f64::from) / n.sqrt();
        let gram = x.dot(&x.t());
        let scale = gram
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|g| g.abs()).sum::<f64>())
            .fold(1.0, f64::max);
        let mut w = x.t().dot(&x) / scale;
        for _ in 0..epochs {
            let y = x.dot(&w.t());
            let decay = match variant {
                OjaVariant::Subspace => y.t().dot(&y).dot(&w),
                OjaVariant::SingleNeuron => {
                    let squares = y.map_axis(Axis(0), |column| column.dot(&column));
                    &w * &squares.insert_axis(Axis(1))
                }
            };
            w = &w + &((y.t().dot(&x) - decay) * (eta / p));
        }
        w
    }

    fn oja(variant: OjaVariant, max_epochs: usize) -> Oja {
        Oja {
            variant,
            max_epochs,
            tolerance: 0.0,
            ..Oja::default()
        }
    }

    #[test]
    fn small_matrix_updates_match_the_direct_rule() {
        for set in [three_patterns(), correlated_patterns()] {
            for variant in OjaVariant::iter() {
                let fit = oja(variant, 3).fit(&set).unwrap();
                let expected = reference(&set, variant, 0.1, 3);
                let w = full_matrix(&fit, &set);
                assert!(
                    w.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-14),
                    "{variant}"
                );
                assert_eq!((fit.epochs, fit.converged), (3, false));
            }
        }
    }

    /// Hand computation for `two_patterns`, `eta = 0.1`, `p = 2`, `N = 25`.
    /// `G = Xi Xi^T / N = [[1, 0.2], [0.2, 1]]` has eigenvalues 1.2 (for
    /// `a + b`) and 0.8 (for `a - b`), and the Gershgorin scale is
    /// `s = 1.2`. The subspace rule keeps `W = Xi^T M Xi / N` and acts on
    /// each eigenvalue: with `f = lambda m`,
    /// `f <- f + (eta / p) lambda f (1 - f^2)`, starting from
    /// `f = lambda / s`. After one epoch:
    /// - `f1 = 1` stays 1, so `m1 = 1 / 1.2`,
    /// - `f2 = 2/3 + 0.05 * 0.8 * (2/3) * (1 - 4/9) = 18.4 / 27`,
    ///   `m2 = f2 / 0.8 = 23 / 27`.
    ///
    /// Pixels 0-14 are `(1, 1)` in both patterns, along `a + b`, so
    /// `w_ij = 2 m1 / 25 = 1/15` there; pixels 15-24 are `(1, -1)`, so
    /// `w_ij = 2 m2 / 25 = 46 / 675`; the cross block is 0.
    #[test]
    fn subspace_rule_after_one_epoch_matches_hand_computation() {
        let w = oja(OjaVariant::Subspace, 1)
            .train(&two_patterns(), Backend::Dense)
            .unwrap()
            .to_dense();
        assert!((w[[0, 1]] - 1.0 / 15.0).abs() < 1e-15, "{}", w[[0, 1]]);
        assert!(
            (w[[15, 16]] - 46.0 / 675.0).abs() < 1e-15,
            "{}",
            w[[15, 16]]
        );
        assert!(w[[0, 15]].abs() < 1e-15, "{}", w[[0, 15]]);
        assert_eq!(w[[3, 3]], 0.0);
    }

    /// The converged subspace rule is the projector
    /// `P = Xi^T (Xi Xi^T)^-1 Xi`: `f = 1`, so `m1 = 1 / 1.2` and
    /// `m2 = 1 / 0.8`, giving `w_ij = 1/15` on pixels 0-14 (diagonal
    /// included before zeroing), 0.1 on pixels 15-24 and 0 across. Each
    /// pattern is mapped to itself, and after zeroing the diagonal its local
    /// field is `(1 - P_ii) xi_i`, so both patterns are stable.
    #[test]
    fn subspace_rule_converges_to_the_projection_rule() {
        let set = two_patterns();
        let fit = Oja::default().fit(&set).unwrap();
        assert!(fit.converged, "{} epochs", fit.epochs);
        let w = full_matrix(&fit, &set);
        assert!((w[[0, 1]] - 1.0 / 15.0).abs() < 1e-6, "{}", w[[0, 1]]);
        assert!((w[[0, 0]] - 1.0 / 15.0).abs() < 1e-6);
        assert!((w[[15, 16]] - 0.1).abs() < 1e-6, "{}", w[[15, 16]]);
        assert!(w[[0, 15]].abs() < 1e-12);

        let weights = Oja::default().train(&set, Backend::Factored).unwrap();
        for xi in set.iter() {
            let h = weights.local_fields(xi.values());
            assert!(
                h.iter()
                    .zip(xi.values())
                    .all(|(h, &x)| h * f64::from(x) > 0.0)
            );
        }
    }

    /// The small eigenvalues of `G` (about 0.16 here) are learned slowly,
    /// at a rate of about `(eta / p) lambda` per epoch, so this set needs
    /// thousands of epochs.
    #[test]
    fn strongly_correlated_patterns_do_not_diverge() {
        let set = correlated_patterns();
        let rule = Oja {
            max_epochs: 20_000,
            ..Oja::default()
        };
        let fit = rule.fit(&set).unwrap();
        assert!(fit.converged, "{} epochs", fit.epochs);
        let w = full_matrix(&fit, &set);
        for xi in set.iter() {
            let x = xi.values().mapv(f64::from);
            let residual = (w.dot(&x) - &x).iter().map(|r| r.abs()).fold(0.0, f64::max);
            assert!(residual < 1e-3, "{residual}");
        }
    }

    #[test]
    fn reports_divergence() {
        let rule = Oja {
            eta: 1e6,
            ..Oja::default()
        };
        let error = rule.fit(&correlated_patterns()).unwrap_err();
        assert!(matches!(error, Error::LearningDiverged { .. }), "{error}");
    }

    #[test]
    fn subspace_rule_keeps_the_weights_symmetric() {
        let w = oja(OjaVariant::Subspace, 50)
            .train(&three_patterns(), Backend::Dense)
            .unwrap()
            .to_dense();
        assert!(w.iter().zip(w.t()).all(|(a, b)| (a - b).abs() < 1e-15));
        assert!(w.diag().iter().all(|&d| d == 0.0));
    }

    /// Every row of the single-neuron rule converges to a unit eigenvector,
    /// so the row norms of the full matrix tend to 1.
    #[test]
    fn single_neuron_rule_normalizes_every_row() {
        let set = three_patterns();
        let rule = Oja {
            variant: OjaVariant::SingleNeuron,
            max_epochs: 5000,
            ..Oja::default()
        };
        let w = full_matrix(&rule.fit(&set).unwrap(), &set);
        for row in w.rows() {
            assert!(
                (row.dot(&row).sqrt() - 1.0).abs() < 1e-3,
                "{}",
                row.dot(&row)
            );
        }
    }

    #[test]
    fn backends_give_the_same_weights() {
        let set = three_patterns();
        for variant in OjaVariant::iter() {
            let rule = oja(variant, 20);
            let dense = rule.train(&set, Backend::Dense).unwrap();
            let factored = rule.train(&set, Backend::Factored).unwrap();
            assert_eq!(factored.backend(), Backend::Factored);
            let (a, b) = (dense.to_dense(), factored.to_dense());
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-15));
        }
    }

    #[test]
    fn parses_a_configuration_with_defaults() {
        let rule: Oja = toml::from_str("variant = \"single-neuron\"\neta = 0.05").unwrap();
        assert_eq!(rule.variant, OjaVariant::SingleNeuron);
        assert_eq!(rule.eta, 0.05);
        assert_eq!(rule.max_epochs, 100_000);
        assert!(toml::from_str::<Oja>("epochs = 3").is_err());
    }

    #[test]
    fn rejects_invalid_hyperparameters() {
        let set = three_patterns();
        for rule in [
            Oja {
                eta: 0.0,
                ..Oja::default()
            },
            Oja {
                eta: f64::NAN,
                ..Oja::default()
            },
            Oja {
                max_epochs: 0,
                ..Oja::default()
            },
            Oja {
                tolerance: -1.0,
                ..Oja::default()
            },
        ] {
            assert!(rule.fit(&set).is_err(), "{rule:?}");
        }
    }
}
