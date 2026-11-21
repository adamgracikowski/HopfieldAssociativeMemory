//! Aggregates of the trials of a grid cell.

use serde::Serialize;

use super::{Cell, FinalClass, TrialResult};
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
    /// Fraction `count / total` with the 95% Wilson score interval, clamped
    /// so it contains the fraction.
    ///
    /// # Panics
    ///
    /// Panics if `total` is 0 or smaller than `count`.
    pub fn wilson(count: usize, total: usize) -> Self {
        Self::check(count, total);
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

impl Proportion {
    /// Fraction `count / total` with both interval ends equal to it.
    ///
    /// # Panics
    ///
    /// Panics if `total` is 0 or smaller than `count`.
    pub fn exact(count: usize, total: usize) -> Self {
        Self::check(count, total);
        let fraction = count as f64 / total as f64;
        Self {
            count,
            fraction,
            lower: fraction,
            upper: fraction,
        }
    }

    /// [Exact](Self::exact) if `deterministic`, otherwise
    /// [Wilson](Self::wilson).
    pub fn of(count: usize, total: usize, deterministic: bool) -> Self {
        if deterministic {
            Self::exact(count, total)
        } else {
            Self::wilson(count, total)
        }
    }

    fn check(count: usize, total: usize) {
        assert!(total > 0 && count <= total, "{count} of {total} trials");
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
    /// Class fractions of `results`, which must not be empty; exact for a
    /// `deterministic` cell.
    fn of(results: &[TrialResult], deterministic: bool) -> Self {
        let share = |class: FinalClass| {
            let count = results.iter().filter(|r| r.class == class).count();
            Proportion::of(count, results.len(), deterministic)
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
    /// Damage level: the fraction of flipped pixels, or of covered rows;
    /// `None` for random initial states.
    pub noise: Option<f64>,
    /// Number of damaged pixels; `None` for random initial states.
    pub flips: Option<usize>,
    /// Number of runs: trials times targets.
    pub runs: usize,
    /// Whether the cell ran a single trial with exact fractions
    /// (synchronous updates of undamaged patterns).
    pub deterministic: bool,
    /// Fractions of the final state classes.
    pub classes: Classes,
    /// Fraction of runs that ended in their target; `None` for random
    /// initial states.
    pub recovered: Option<Proportion>,
    /// Fraction of runs with [`TrialResult::nearest_target`]; `None` for
    /// random initial states.
    pub nearest_target: Option<Proportion>,
    /// Applied steps.
    pub steps: Moments,
    /// Energy of the final state.
    pub energy: Moments,
    /// Overlap of the final state with the target; `None` for random
    /// initial states.
    pub final_overlap: Option<Moments>,
    /// Hamming distance of the final state to the target; `None` for random
    /// initial states.
    pub final_distance: Option<Moments>,
}

impl CellSummary {
    /// Aggregates `results`, all from cell `cell`, which are not empty.
    pub fn new(dataset: String, learning: String, cell: Cell, results: &[TrialResult]) -> Self {
        let first = results.first().expect("a cell has at least one run");
        let deterministic = cell.is_deterministic(first.trial.damaged_pixels());
        let share = |flags: Vec<bool>| {
            (!flags.is_empty()).then(|| {
                let count = flags.iter().filter(|&&flag| flag).count();
                Proportion::of(count, flags.len(), deterministic)
            })
        };
        Self {
            dataset,
            learning,
            mode: cell.mode,
            noise: cell.noise,
            flips: cell.noise.map(|_| first.trial.damaged_pixels()),
            runs: results.len(),
            deterministic,
            classes: Classes::of(results, deterministic),
            recovered: share(results.iter().filter_map(|r| r.recovered).collect()),
            nearest_target: share(results.iter().filter_map(|r| r.nearest_target()).collect()),
            steps: Moments::of(results.iter().map(|r| r.steps as f64))
                .expect("a cell has at least one run"),
            energy: Moments::of(results.iter().map(|r| r.energy))
                .expect("a cell has at least one run"),
            final_overlap: Moments::of(results.iter().filter_map(|r| r.final_overlap)),
            final_distance: Moments::of(
                results
                    .iter()
                    .filter_map(|r| r.final_distance.map(|d| d as f64)),
            ),
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

    /// Target 1, `N = 49`; final `m` with the target, nearest `(pattern, m)`:
    /// - `1`, `(1, 1)`: recovered,
    /// - `41/49`, `(0, 45/49)`: another pattern closer,
    /// - `-45/49`, `(1, -45/49)`: the negated target,
    /// - `39/49`, `(0, 39/49)`: a tie, nearest the target.
    ///
    /// Final distances `0, 4, 47, 5`: mean 14, squared deviations
    /// `196 + 100 + 1089 + 81 = 1466`, deviation `sqrt(1466 / 3) = 22.105806`.
    #[test]
    fn summarizes_the_nearest_target_and_the_final_distance() {
        use crate::dynamics::Termination;
        use crate::experiment::{Damage, Trial};
        use crate::run::Nearest;
        use crate::seed::Seed;

        let result = |pattern, overlap: f64, m: f64, distance| TrialResult {
            trial: Trial {
                cell: 0,
                trial: 0,
                target: Some(1),
                damage: Some(Damage::Flip { pixels: 5 }),
                seed: Seed(0),
            },
            steps: 1,
            changes: 1,
            termination: Termination::FixedPoint,
            energy: -1.0,
            class: if distance == 0 {
                FinalClass::Pattern
            } else {
                FinalClass::Spurious
            },
            recovered: Some(distance == 0),
            initial_distance: Some(5),
            final_distance: Some(distance),
            final_overlap: Some(m / 49.0),
            nearest: Nearest {
                pattern,
                overlap: overlap / 49.0,
            },
        };
        let results = [
            result(1, 49.0, 49.0, 0),
            result(0, 45.0, 41.0, 4),
            result(1, -45.0, -45.0, 47),
            result(0, 39.0, 39.0, 5),
        ];
        let cell = Cell {
            dataset: 0,
            learning: 0,
            mode: Mode::Asynchronous,
            noise: Some(0.1),
        };
        let summary = CellSummary::new("set".to_owned(), "hebb".to_owned(), cell, &results);
        assert_eq!(summary.recovered, Some(Proportion::wilson(1, 4)));
        assert_eq!(summary.nearest_target, Some(Proportion::wilson(2, 4)));
        let distance = summary.final_distance.unwrap();
        assert_eq!(distance.mean, 14.0);
        assert!(
            (distance.std.unwrap() - 22.105806).abs() < 1e-6,
            "{distance:?}"
        );
    }

    #[test]
    fn exact_proportion_has_no_width() {
        let exact = Proportion::of(3, 5, true);
        assert_eq!((exact.fraction, exact.lower, exact.upper), (0.6, 0.6, 0.6));
        assert_eq!(Proportion::of(3, 5, false), Proportion::wilson(3, 5));
    }
}
