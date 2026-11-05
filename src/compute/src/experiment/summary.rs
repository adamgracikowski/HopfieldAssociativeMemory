//! Aggregates of the trials of a grid cell.

use serde::Serialize;

use super::{FinalClass, TrialResult};
use crate::dynamics::Mode;

/// Quantile of the standard normal distribution for a 95% interval.
const Z_95: f64 = 1.96;

/// A fraction of trials with its 95% Wilson score interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Proportion {
    /// Number of trials with the property.
    pub count: usize,
    /// Fraction `k / n` of the trials.
    pub fraction: f64,
    /// Lower end of the interval.
    pub lower: f64,
    /// Upper end of the interval.
    pub upper: f64,
}

impl Proportion {
    /// Fraction `count / total` with the Wilson score interval: for
    /// `p = k / n` and `z = 1.96` the centre is
    /// `(p + z^2 / 2n) / (1 + z^2 / n)` and the half-width
    /// `z / (1 + z^2 / n) * sqrt(p (1 - p) / n + z^2 / 4n^2)`. Unlike the
    /// normal approximation it stays within `[0, 1]` and is not empty for
    /// `k = 0` or `k = n`. The interval always contains `p`, with ends 0 for
    /// `k = 0` and 1 for `k = n`; the computed ends are clamped to `[0, p]`
    /// and `[p, 1]`, because rounding can move them slightly past `p` (e.g.
    /// an upper end of `1 - 1e-16` for `k = n = 100`).
    ///
    /// # Panics
    ///
    /// Panics if `total` is 0 or smaller than `count`.
    pub fn wilson(count: usize, total: usize) -> Self {
        assert!(total > 0 && count <= total, "{count} of {total} trials");
        let n = total as f64;
        let p = count as f64 / n;
        let z2 = Z_95 * Z_95;
        let denominator = 1.0 + z2 / n;
        let centre = (p + z2 / (2.0 * n)) / denominator;
        let half = Z_95 / denominator * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
        Self {
            count,
            fraction: p,
            lower: (centre - half).clamp(0.0, p),
            upper: (centre + half).clamp(p, 1.0),
        }
    }
}

/// Mean and sample standard deviation of a quantity over trials.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Moments {
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (divided by `n - 1`); `None` for a single
    /// value.
    pub std: Option<f64>,
}

impl Moments {
    /// Moments of `values`, or `None` if there are none.
    pub fn of(values: impl IntoIterator<Item = f64>) -> Option<Self> {
        let values: Vec<f64> = values.into_iter().collect();
        if values.is_empty() {
            return None;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = (values.len() > 1).then(|| {
            let squares: f64 = values.iter().map(|x| (x - mean).powi(2)).sum();
            (squares / (n - 1.0)).sqrt()
        });
        Some(Self { mean, std })
    }
}

/// Fractions of trials in every [`FinalClass`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Classes {
    /// Ended in a stored pattern.
    pub pattern: Proportion,
    /// Ended in a negated stored pattern.
    pub negation: Proportion,
    /// Ended in another fixed point.
    pub spurious: Proportion,
    /// Ended in a cycle of length 2.
    pub cycle: Proportion,
    /// Reached the step limit.
    pub unconverged: Proportion,
}

impl Classes {
    /// Class fractions of `results`, which must not be empty.
    fn of(results: &[TrialResult]) -> Self {
        let share = |class: FinalClass| {
            let count = results.iter().filter(|r| r.class == class).count();
            Proportion::wilson(count, results.len())
        };
        Self {
            pattern: share(FinalClass::Pattern),
            negation: share(FinalClass::Negation),
            spurious: share(FinalClass::Spurious),
            cycle: share(FinalClass::Cycle),
            unconverged: share(FinalClass::Unconverged),
        }
    }
}

/// Aggregates of all trials of one grid cell.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CellSummary {
    /// Dataset name.
    pub dataset: String,
    /// Learning rule name.
    pub learning: String,
    /// Update mode.
    pub mode: Mode,
    /// Fraction of flipped pixels; `None` for random initial states.
    pub noise: Option<f64>,
    /// Number of flipped pixels; `None` for random initial states.
    pub flips: Option<usize>,
    /// Number of runs: trials times targets.
    pub runs: usize,
    /// Fractions of the final state classes.
    pub classes: Classes,
    /// Fraction of runs that ended in their target; `None` for random
    /// initial states.
    pub recovered: Option<Proportion>,
    /// Applied steps.
    pub steps: Moments,
    /// Energy of the final state.
    pub energy: Moments,
    /// Overlap of the final state with the target; `None` for random
    /// initial states.
    pub final_overlap: Option<Moments>,
}

impl CellSummary {
    /// Aggregates `results`, all from the same cell, which are not empty.
    pub fn new(
        dataset: String,
        learning: String,
        mode: Mode,
        noise: Option<f64>,
        results: &[TrialResult],
    ) -> Self {
        let first = results.first().expect("a cell has at least one run");
        let recovered: Vec<bool> = results.iter().filter_map(|r| r.recovered).collect();
        Self {
            dataset,
            learning,
            mode,
            noise,
            flips: noise.map(|_| first.trial.flips),
            runs: results.len(),
            classes: Classes::of(results),
            recovered: (!recovered.is_empty()).then(|| {
                Proportion::wilson(recovered.iter().filter(|&&r| r).count(), recovered.len())
            }),
            steps: Moments::of(results.iter().map(|r| r.steps as f64))
                .expect("a cell has at least one run"),
            energy: Moments::of(results.iter().map(|r| r.energy))
                .expect("a cell has at least one run"),
            final_overlap: Moments::of(results.iter().filter_map(|r| r.final_overlap)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `k = 8`, `n = 10`: `p = 0.8`, `1 + z^2 / n = 1.38416`, centre
    /// `(0.8 + 0.19208) / 1.38416 = 0.716738`, half-width
    /// `1.96 / 1.38416 * sqrt(0.016 + 0.009604) = 0.226581`, so the interval
    /// is `(0.490157, 0.943319)`.
    #[test]
    fn wilson_interval_of_eight_in_ten() {
        let proportion = Proportion::wilson(8, 10);
        assert_eq!((proportion.count, proportion.fraction), (8, 0.8));
        assert!((proportion.lower - 0.490157).abs() < 1e-6, "{proportion:?}");
        assert!((proportion.upper - 0.943319).abs() < 1e-6, "{proportion:?}");
    }

    /// `k = 0`, `n = 5`: centre and half-width are both
    /// `0.38416 / 1.76832 = 0.217246`, so the interval is `(0, 0.434491)`;
    /// the lower end, `-3e-17` after rounding, is clamped to 0.
    #[test]
    fn wilson_interval_of_none() {
        let proportion = Proportion::wilson(0, 5);
        assert_eq!((proportion.fraction, proportion.lower), (0.0, 0.0));
        assert!((proportion.upper - 0.434491).abs() < 1e-6, "{proportion:?}");
        let all = Proportion::wilson(5, 5);
        assert_eq!(all.upper, 1.0);
        assert!((all.lower - (1.0 - 0.434491)).abs() < 1e-6, "{all:?}");
    }

    /// For `k = 0` and `k = n` the interval ends at exactly 0 and 1 for any
    /// `n`, even where rounding of the formula misses them, e.g. an upper
    /// end of `1 - 1e-16` for `k = n = 100`.
    #[test]
    fn wilson_interval_contains_the_fraction() {
        for n in 1..=500 {
            assert_eq!(Proportion::wilson(0, n).lower, 0.0, "n = {n}");
            assert_eq!(Proportion::wilson(n, n).upper, 1.0, "n = {n}");
            for k in [n / 3, n / 2] {
                let proportion = Proportion::wilson(k, n);
                assert!(proportion.lower <= proportion.fraction, "{k} of {n}");
                assert!(proportion.fraction <= proportion.upper, "{k} of {n}");
            }
        }
    }

    /// `1, 2, 3, 4`: mean 2.5, squared deviations `2.25 + 0.25 + 0.25 +
    /// 2.25 = 5`, sample variance `5 / 3`, deviation `1.290994`.
    #[test]
    fn moments_of_a_sequence() {
        let moments = Moments::of([1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(moments.mean, 2.5);
        assert!((moments.std.unwrap() - 1.290994).abs() < 1e-6);
        assert_eq!(
            Moments::of([3.0]),
            Some(Moments {
                mean: 3.0,
                std: None
            })
        );
        assert_eq!(Moments::of([]), None);
    }
}
