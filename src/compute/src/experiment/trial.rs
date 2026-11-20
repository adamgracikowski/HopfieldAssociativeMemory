//! A single trial of an experiment.

use super::{Damage, FinalClass};
use crate::dynamics::{Dynamics, NoHistory, Termination};
use crate::patterns::Pattern;
use crate::run::{Input, Nearest};
use crate::seed::Seed;
use crate::weights::Weights;
use crate::{PatternSet, Result};

/// One run of a trained network from one initial state; `recall` with this
/// seed and the same settings repeats it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trial {
    /// Index of the grid cell.
    pub cell: usize,
    /// Number of the trial within the cell, from 0.
    pub trial: usize,
    /// Stored pattern the initial state is made from; `None` for a random
    /// state.
    pub target: Option<usize>,
    /// Damage of the target; `None` for a random state.
    pub damage: Option<Damage>,
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
    /// Whether the target is among the patterns closest to the final state:
    /// its overlap is positive and at least the largest `|m|` (ties count).
    /// `None` for a random initial state.
    pub fn nearest_target(&self) -> Option<bool> {
        self.final_overlap
            .map(|m| m > 0.0 && m >= self.nearest.overlap.abs())
    }
}

impl Trial {
    /// The initial state settings of this trial.
    pub fn input(&self) -> Input {
        match (self.target, self.damage) {
            (Some(index), Some(damage)) => damage.input(index),
            _ => Input::Random {},
        }
    }

    /// Number of damaged pixels of the target; 0 for a random state.
    pub fn damaged_pixels(&self) -> usize {
        self.damage.map_or(0, |damage| damage.pixels())
    }

    /// Runs the network storing `set` without recording frames.
    pub fn run(
        &self,
        weights: &Weights,
        set: &PatternSet,
        dynamics: &Dynamics,
    ) -> Result<TrialResult> {
        self.run_with_state(weights, set, dynamics)
            .map(|(result, _)| result)
    }

    /// Like [`run`](Self::run), but also returns the final state.
    pub fn run_with_state(
        &self,
        weights: &Weights,
        set: &PatternSet,
        dynamics: &Dynamics,
    ) -> Result<(TrialResult, Pattern)> {
        let initial = self.input().initial(set, &mut self.seed.input_rng())?;
        let outcome = dynamics.run(
            weights,
            initial.state.view(),
            &mut self.seed.dynamics_rng(),
            &mut NoHistory,
        )?;
        let target = self.target.map(|mu| set.pattern(mu));
        let overlaps: Vec<f64> = set.iter().map(|xi| outcome.state.overlap(&xi)).collect();
        let result = TrialResult {
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
        };
        Ok((result, outcome.state))
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
            damage: Some(Damage::Flip { pixels: flips }),
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

    /// A final state with overlap `0.5` with both patterns: `Nearest::of`
    /// keeps pattern 0, but target 1 ties with it and counts as nearest.
    #[test]
    fn a_target_tied_for_the_largest_overlap_is_nearest() {
        let mut result = run(array![[1, 1]], 0);
        result.trial.target = Some(1);
        result.final_overlap = Some(0.5);
        result.nearest = Nearest {
            pattern: 0,
            overlap: -0.5,
        };
        assert_eq!(result.nearest_target(), Some(true));
        result.final_overlap = Some(0.25);
        assert_eq!(result.nearest_target(), Some(false));
        result.final_overlap = None;
        assert_eq!(result.nearest_target(), None);
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
