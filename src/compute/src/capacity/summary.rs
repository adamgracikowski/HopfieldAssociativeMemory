//! Aggregates of the pattern sets of one source, size and rule.

use serde::Serialize;

use super::CapacityRow;
use crate::experiment::{Moments, Proportion};

/// Stability of random patterns under the Hebbian rule predicted by the
/// signal-to-noise argument (MacKay 2003, eqs. 42.18-42.22): the aligned
/// field of a neuron is approximately normal with mean `N` and variance
/// `N p`, so a neuron is unstable with probability
/// `Phi(-sqrt(N / p))`, and a pattern of `N` independent neurons is stable
/// with probability `(1 - Phi(-sqrt(N / p)))^N`. Large-`N` approximation;
/// with the zero diagonal, measured sets are slightly more stable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Theory {
    /// Probability that one neuron of a pattern is unstable.
    pub bit_error: f64,
    /// Probability that a whole pattern is stable.
    pub pattern_stable: f64,
}

impl Theory {
    /// The prediction for `patterns` random patterns of `neurons` neurons;
    /// `Phi(x) = erfc(-x / sqrt 2) / 2`, via `libm` for platform-independent
    /// results.
    pub fn new(neurons: usize, patterns: usize) -> Self {
        let ratio = (neurons as f64 / patterns as f64).sqrt();
        let bit_error = 0.5 * libm::erfc(ratio / std::f64::consts::SQRT_2);
        Self {
            bit_error,
            pattern_stable: libm::pow(1.0 - bit_error, neurons as f64),
        }
    }
}

/// Aggregates of the pattern sets of one source, number of patterns and
/// learning rule.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CapacitySummary {
    /// Dataset name, or `random`.
    pub source: String,
    /// Learning rule name.
    pub learning: String,
    /// Number of neurons `N`.
    pub neurons: usize,
    /// Number of patterns `p`.
    pub patterns: usize,
    /// Load `p / N`.
    pub load: f64,
    /// Number of pattern sets (trials).
    pub trials: usize,
    /// Whether the sets are fixed (a dataset prefix, a single trial): the
    /// fractions are exact.
    pub deterministic: bool,
    /// Fraction `stable / p` of stable patterns of a set.
    pub stable_fraction: Moments,
    /// Fraction of sets whose patterns are all stable, with the 95% Wilson
    /// interval (exact for a deterministic source).
    pub all_stable: Proportion,
    /// Fraction `unstable neurons / (p N)` of neurons of a set that would
    /// change.
    pub bit_errors: Moments,
    /// Prediction for random patterns with the Hebbian rule; `None`
    /// otherwise.
    pub theory: Option<Theory>,
}

impl CapacitySummary {
    /// Aggregates `rows`, which are not empty and share the source, the
    /// number of patterns and the rule; with `theory` the prediction for
    /// random Hebbian sets is added.
    pub fn new(source: String, learning: String, rows: &[&CapacityRow], theory: bool) -> Self {
        let first = rows.first().expect("a group has at least one set");
        let neurons = first.neurons();
        let patterns = first.task.patterns;
        let deterministic = first.task.seed.is_none();
        let all = rows
            .iter()
            .filter(|row| row.stability.stable == patterns)
            .count();
        Self {
            source,
            learning,
            neurons,
            patterns,
            load: patterns as f64 / neurons as f64,
            trials: rows.len(),
            deterministic,
            stable_fraction: Moments::of(
                rows.iter()
                    .map(|row| row.stability.stable as f64 / patterns as f64),
            )
            .expect("a group has at least one set"),
            all_stable: Proportion::of(all, rows.len(), deterministic),
            bit_errors: Moments::of(
                rows.iter()
                    .map(|row| row.unstable_neurons() as f64 / (patterns * neurons) as f64),
            )
            .expect("a group has at least one set"),
            theory: theory.then(|| Theory::new(neurons, patterns)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `N = 100`, `p = 10`: `Phi(-sqrt(N / p)) = erfc(2.236068) / 2 =
    /// 7.827e-4` (Python's `math.erfc`), so a pattern is stable with
    /// probability `(1 - 7.827e-4)^100 = 0.924686`.
    #[test]
    fn theory_of_ten_patterns_of_a_hundred_neurons() {
        let theory = Theory::new(100, 10);
        assert!((theory.bit_error - 7.827011e-4).abs() < 1e-9, "{theory:?}");
        assert!(
            (theory.pattern_stable - 0.924686).abs() < 1e-6,
            "{theory:?}"
        );
    }

    /// Three sets of `p = 2`, `N = 4` with 2, 1, 0 stable patterns and 0, 3,
    /// 3 unstable neurons:
    /// - stable fractions `1, 0.5, 0`: mean 0.5, std `sqrt(0.5 / 2) = 0.5`,
    /// - unstable neurons per `p N = 8`: `0, 0.375, 0.375`, mean 0.25, std
    ///   `sqrt((0.0625 + 0.015625 + 0.015625) / 2) = sqrt(0.046875)`,
    /// - all patterns stable in one set of three.
    #[test]
    fn aggregates_the_sets_of_a_group() {
        use crate::capacity::{CapacityRow, Task};
        use crate::dynamics::Stability;
        use crate::run::TrainingInfo;
        use crate::weights::Backend;

        let rows: Vec<CapacityRow> = [(2, vec![0, 0]), (1, vec![0, 3]), (0, vec![1, 2])]
            .into_iter()
            .enumerate()
            .map(|(trial, (stable, unstable_neurons))| CapacityRow {
                task: Task::random(1, 4, 2, trial),
                learning: 0,
                training: TrainingInfo {
                    backend: Backend::Dense,
                    neurons: 4,
                    epochs: 0,
                    converged: true,
                },
                stability: Stability {
                    stable,
                    unstable_neurons,
                },
            })
            .collect();
        let refs: Vec<&CapacityRow> = rows.iter().collect();
        let summary = CapacitySummary::new("random".into(), "hebb".into(), &refs, true);
        assert_eq!(
            (summary.neurons, summary.patterns, summary.trials),
            (4, 2, 3)
        );
        assert_eq!(summary.load, 0.5);
        assert!(!summary.deterministic);
        assert!((summary.stable_fraction.mean - 0.5).abs() < 1e-12);
        assert!((summary.stable_fraction.std.unwrap() - 0.5).abs() < 1e-12);
        assert!((summary.bit_errors.mean - 0.25).abs() < 1e-12);
        assert!((summary.bit_errors.std.unwrap() - 0.046875_f64.sqrt()).abs() < 1e-12);
        assert_eq!(summary.all_stable.count, 1);
        assert!((summary.all_stable.fraction - 1.0 / 3.0).abs() < 1e-12);
        assert!(summary.all_stable.lower < 1.0 / 3.0 && summary.all_stable.upper > 1.0 / 3.0);
        assert_eq!(summary.theory, Some(Theory::new(4, 2)));
    }

    #[test]
    fn theory_bit_error_grows_with_the_load() {
        let errors: Vec<f64> = [5, 10, 20, 40]
            .iter()
            .map(|&p| Theory::new(100, p).bit_error)
            .collect();
        assert!(
            errors.windows(2).all(|pair| pair[0] < pair[1]),
            "{errors:?}"
        );
    }
}
