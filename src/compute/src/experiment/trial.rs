//! A single trial of an experiment.

use super::FinalClass;
use crate::dynamics::{Dynamics, NoHistory, Termination};
use crate::run::{Input, Nearest};
use crate::seed::Seed;
use crate::weights::Weights;
use crate::{PatternSet, Result};

/// One run of a trained network from one initial state.
///
/// The initial state is built by [`Input`] from the input stream of `seed`
/// and the dynamics use its dynamics stream, exactly as in a `recall` run.
/// So `recall` repeats the trial with this seed, the experiment's dataset,
/// `registry`, `backend` and `[learning]` table of the rule, `[dynamics]`
/// with the cell's `mode` and the experiment's `zero_field`, `order` and
/// `max_steps`, and the input `pattern` with `index = target` and `flips`
/// (or `random`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trial {
    /// Index of the grid cell.
    pub cell: usize,
    /// Number of the trial within the cell, from 0.
    pub trial: usize,
    /// Stored pattern the initial state is made from; `None` for a random
    /// state.
    pub target: Option<usize>,
    /// Number of flipped pixels of the target.
    pub flips: usize,
    /// Seed of the trial.
    pub seed: Seed,
}

/// Result of a trial: one row of the trial table.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialResult {
    /// The trial.
    pub trial: Trial,
    /// Applied steps (synchronous updates or asynchronous sweeps).
    pub steps: usize,
    /// Total number of neuron changes.
    pub changes: usize,
    /// Why the run stopped.
    pub termination: Termination,
    /// Energy of the final state.
    pub energy: f64,
    /// Class of the final state.
    pub class: FinalClass,
    /// Whether the run ended in a fixed point equal to the target (a state
    /// of a two-cycle or at the step limit does not count).
    pub recovered: Option<bool>,
    /// Hamming distance of the initial state to the target.
    pub initial_distance: Option<usize>,
    /// Hamming distance of the final state to the target.
    pub final_distance: Option<usize>,
    /// Overlap of the final state with the target.
    pub final_overlap: Option<f64>,
    /// Stored pattern closest to the final state.
    pub nearest: Nearest,
}

impl TrialResult {
    /// Whether the stored pattern closest to the final state (the largest
    /// `|m|`, the first one on ties) is the target itself and not its
    /// negation; `None` for a random initial state. Unlike
    /// [`recovered`](Self::recovered) it holds for a run that ends near an
    /// unstable target, in any final class.
    pub fn nearest_target(&self) -> Option<bool> {
        self.trial
            .target
            .map(|mu| self.nearest.pattern == mu && self.nearest.overlap > 0.0)
    }
}

impl Trial {
    /// The initial state settings of this trial.
    pub fn input(&self) -> Input {
        match self.target {
            Some(index) => Input::Pattern {
                index,
                flips: self.flips,
            },
            None => Input::Random {},
        }
    }

    /// Runs the network with `weights`, which stores `set`, under
    /// `dynamics`, without recording frames.
    pub fn run(
        &self,
        weights: &Weights,
        set: &PatternSet,
        dynamics: &Dynamics,
    ) -> Result<TrialResult> {
        let initial = self.input().initial(set, &mut self.seed.input_rng())?;
        let outcome = dynamics.run(
            weights,
            initial.state.view(),
            &mut self.seed.dynamics_rng(),
            &mut NoHistory,
        )?;
        let target = self.target.map(|mu| set.pattern(mu));
        let overlaps: Vec<f64> = set.iter().map(|xi| outcome.state.overlap(&xi)).collect();
        Ok(TrialResult {
            trial: *self,
            steps: outcome.steps,
            changes: outcome.changes,
            termination: outcome.termination,
            energy: outcome.energy,
            class: FinalClass::of(&outcome, set),
            recovered: target
                .map(|xi| outcome.termination == Termination::FixedPoint && outcome.state == xi),
            initial_distance: target.map(|xi| initial.state.hamming(&xi)),
            final_distance: target.map(|xi| outcome.state.hamming(&xi)),
            final_overlap: self.target.map(|mu| overlaps[mu]),
            nearest: Nearest::of(&overlaps),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::Mode;
    use crate::weights::DenseWeights;
    use ndarray::array;

    /// Two neurons with `w_01 = w_10 = 0.5` (Hebbian weights of `(1, 1)`).
    fn weights() -> Weights {
        Weights::from(DenseWeights::new(array![[0.0, 0.5], [0.5, 0.0]]).unwrap())
    }

    fn run(stored: ndarray::Array2<i8>, flips: usize) -> TrialResult {
        let set = PatternSet::new(2, 1, stored).unwrap();
        let trial = Trial {
            cell: 0,
            trial: 0,
            target: Some(0),
            flips,
            seed: Seed(0),
        };
        let dynamics = Dynamics {
            mode: Mode::Synchronous,
            ..Dynamics::default()
        };
        trial.run(&weights(), &set, &dynamics).unwrap()
    }

    /// Target `(1, -1)` with both pixels flipped starts at `(-1, 1)`:
    /// `h = (0.5, -0.5)` gives `(1, -1)`, then `h = (-0.5, 0.5)` gives
    /// `(-1, 1)` again, a cycle of length 2 found after one step whose
    /// final state equals the target. It is not a fixed point, so the run
    /// does not count as recovered, but the target is the nearest pattern.
    #[test]
    fn a_two_cycle_through_the_target_is_not_recovered() {
        let result = run(array![[1, -1]], 2);
        assert_eq!(result.termination, Termination::TwoCycle);
        assert_eq!(result.class, FinalClass::Cycle);
        assert_eq!(result.final_distance, Some(0));
        assert_eq!(result.recovered, Some(false));
        assert_eq!(result.nearest_target(), Some(true));
    }

    /// Target `(1, 1)` with both pixels flipped starts at `(-1, -1)`:
    /// `h = (-0.5, -0.5)` keeps it, a fixed point equal to the negated
    /// target with `m = -1`, which is not the target.
    #[test]
    fn the_negated_target_is_not_the_nearest_target() {
        let result = run(array![[1, 1]], 2);
        assert_eq!(result.class, FinalClass::Negation);
        assert_eq!(
            result.nearest,
            Nearest {
                pattern: 0,
                overlap: -1.0
            }
        );
        assert_eq!(result.nearest_target(), Some(false));
    }

    /// `(1, 1)` has `h = (0.5, 0.5)`: a fixed point equal to the target.
    #[test]
    fn a_fixed_point_equal_to_the_target_is_recovered() {
        let result = run(array![[1, 1]], 0);
        assert_eq!(result.termination, Termination::FixedPoint);
        assert_eq!(result.class, FinalClass::Pattern);
        assert_eq!(result.recovered, Some(true));
        assert_eq!(result.nearest_target(), Some(true));
    }
}
