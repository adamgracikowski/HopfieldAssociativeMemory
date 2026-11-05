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
/// and the dynamics use its dynamics stream, exactly as in a `recall` run,
/// so `recall` with this seed, the same dataset, rule and mode, and input
/// `pattern` with `index = target` and `flips` (or `random`) repeats the
/// trial.
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
    /// Whether the final state equals the target.
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
            recovered: target.map(|xi| outcome.state == xi),
            initial_distance: target.map(|xi| initial.state.hamming(&xi)),
            final_distance: target.map(|xi| outcome.state.hamming(&xi)),
            final_overlap: self.target.map(|mu| overlaps[mu]),
            nearest: Nearest::of(&overlaps),
        })
    }
}
