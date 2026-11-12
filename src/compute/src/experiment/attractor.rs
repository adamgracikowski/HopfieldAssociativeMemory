//! Distinct final states of the runs of a grid cell.

use std::collections::BTreeMap;

use ndarray::Array1;

use super::{FinalClass, Mixture, Trial, TrialResult};
use crate::PatternSet;
use crate::dynamics::{Termination, ZeroField};
use crate::patterns::{BipolarVector, Pattern};
use crate::run::Nearest;
use crate::weights::{WeightMatrix, Weights};

/// One distinct final state of the runs of a cell, with how often the runs
/// ended in it.
///
/// A cycle of length 2 is one attractor, represented by the lexicographically
/// smaller of its two states (`-1 < 1`, pixels in row-major order), so runs
/// that stop in either state of the same cycle count together.
#[derive(Debug, Clone, PartialEq)]
pub struct Attractor {
    /// Index of the grid cell.
    pub cell: usize,
    /// Position among the attractors of the cell, from 0, by `count` from
    /// the largest, then by the first run in the order of the trials.
    pub rank: usize,
    /// Number of runs that ended in this state.
    pub count: usize,
    /// Fraction `count / runs` of the runs of the cell.
    pub fraction: f64,
    /// Class of the final state.
    pub class: FinalClass,
    /// Why the runs stopped.
    pub termination: Termination,
    /// Energy `-1/2 x^T W x` of the state, computed afresh from the
    /// weights, so it does not depend on the run that reached it.
    pub energy: f64,
    /// Overlaps `m` of the state with every stored pattern.
    pub overlaps: Vec<f64>,
    /// Stored pattern with the largest `|m|`.
    pub nearest: Nearest,
    /// The closest mixture of stored patterns (see [`Mixture::closest`]);
    /// `None` for fewer than three patterns.
    pub mixture: Option<Mixture>,
    /// The first run that ended in this state, in the order of the trials;
    /// its seed repeats the run with `recall`.
    pub first: Trial,
    /// The state.
    pub state: Pattern,
}

impl Attractor {
    /// The distinct final states of `runs`, all of one cell, each a trial
    /// result with its final state, in the order of the trials. The network
    /// has `weights`, which store `set`, and the dynamics use `zero_field`,
    /// which gives the second state of a cycle of length 2 (one synchronous
    /// step from the final state).
    ///
    /// Runs are grouped by termination and state, so a state reached as a
    /// fixed point and at the step limit gives two attractors. The result
    /// depends only on `runs` and their order, not on the number of threads
    /// that computed them.
    ///
    /// # Panics
    ///
    /// Panics if `runs` is empty.
    pub fn collect(
        runs: &[(TrialResult, Pattern)],
        set: &PatternSet,
        weights: &Weights,
        zero_field: ZeroField,
    ) -> Vec<Self> {
        assert!(!runs.is_empty(), "a cell has at least one run");
        let mut groups: BTreeMap<(&'static str, Vec<i8>), (usize, usize, Pattern)> =
            BTreeMap::new();
        for (index, (result, state)) in runs.iter().enumerate() {
            let state = Self::canonical(state, result.termination, weights, zero_field);
            let key = (result.termination.into(), state.values().to_vec());
            groups
                .entry(key)
                .and_modify(|(count, _, _)| *count += 1)
                .or_insert((1, index, state));
        }
        let mut groups: Vec<(usize, usize, Pattern)> = groups.into_values().collect();
        groups.sort_by_key(|&(count, first, _)| (std::cmp::Reverse(count), first));
        groups
            .into_iter()
            .enumerate()
            .map(|(rank, (count, first, state))| {
                let result = &runs[first].0;
                let overlaps: Vec<f64> = set.iter().map(|xi| state.overlap(&xi)).collect();
                Self {
                    cell: result.trial.cell,
                    rank,
                    count,
                    fraction: count as f64 / runs.len() as f64,
                    class: result.class,
                    termination: result.termination,
                    energy: weights.energy(state.values()),
                    nearest: Nearest::of(&overlaps),
                    mixture: Mixture::closest(state.view(), &overlaps, set),
                    overlaps,
                    first: result.trial,
                    state,
                }
            })
            .collect()
    }

    /// `state` itself, or for a cycle of length 2 the smaller of `state`
    /// and its synchronous successor `sgn(W state)`.
    fn canonical(
        state: &Pattern,
        termination: Termination,
        weights: &Weights,
        zero_field: ZeroField,
    ) -> Pattern {
        if termination != Termination::TwoCycle {
            return state.clone();
        }
        let fields = weights.local_fields(state.values());
        let successor: Array1<i8> = state
            .values()
            .iter()
            .zip(&fields)
            .map(|(&x, &h)| zero_field.activate(h, x))
            .collect();
        if successor.iter().lt(state.values().iter()) {
            BipolarVector::new_unchecked(successor)
        } else {
            state.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{Dynamics, Mode};
    use crate::experiment::Damage;
    use crate::seed::Seed;
    use crate::weights::DenseWeights;
    use ndarray::array;

    /// Two neurons with `w_01 = w_10 = 0.5` (Hebbian weights of `(1, 1)`).
    fn weights() -> Weights {
        Weights::from(DenseWeights::new(array![[0.0, 0.5], [0.5, 0.0]]).unwrap())
    }

    fn set() -> PatternSet {
        PatternSet::new(2, 1, array![[1, 1]]).unwrap()
    }

    /// Runs the network from target `(1, 1)` with `flips` flipped pixels
    /// and `seed`, synchronously.
    fn run(flips: usize, seed: u64, trial: usize) -> (TrialResult, Pattern) {
        let trial = Trial {
            cell: 3,
            trial,
            target: Some(0),
            damage: Some(Damage::Flip { pixels: flips }),
            seed: Seed(seed),
        };
        let dynamics = Dynamics {
            mode: Mode::Synchronous,
            ..Dynamics::default()
        };
        trial.run_with_state(&weights(), &set(), &dynamics).unwrap()
    }

    /// From `(1, 1)` no pixel changes; from `(-1, -1)` neither (`h = (-0.5,
    /// -0.5)`), a fixed point equal to the negation. With one flipped pixel,
    /// `(1, -1)` has `h = (-0.5, 0.5)` and goes to `(-1, 1)` and back: a
    /// cycle whose final state is either `(-1, 1)` or `(1, -1)`, depending
    /// on the pixel, both represented by the smaller `(-1, 1)`.
    #[test]
    fn groups_runs_by_final_state() {
        let mut runs = vec![run(0, 0, 0), run(2, 0, 0)];
        let cycles: Vec<_> = (0..8).map(|seed| run(1, seed, 1)).collect();
        let finals: Vec<Vec<i8>> = cycles.iter().map(|(_, s)| s.values().to_vec()).collect();
        assert!(finals.contains(&vec![1, -1]) && finals.contains(&vec![-1, 1]));
        runs.extend(cycles);
        runs.push(run(0, 1, 2));

        let attractors = Attractor::collect(&runs, &set(), &weights(), ZeroField::Keep);
        let summary: Vec<_> = attractors
            .iter()
            .map(|a| (a.rank, a.count, a.class, a.state.values().to_vec()))
            .collect();
        assert_eq!(
            summary,
            [
                (0, 8, FinalClass::Cycle, vec![-1, 1]),
                (1, 2, FinalClass::Pattern, vec![1, 1]),
                (2, 1, FinalClass::Negation, vec![-1, -1]),
            ]
        );
        assert_eq!(
            attractors.iter().map(|a| a.count).sum::<usize>(),
            runs.len()
        );
        assert_eq!(attractors[0].first.trial, 1);
        assert_eq!(attractors[1].first, runs[0].0.trial);
        assert_eq!(attractors[1].fraction, 2.0 / 11.0);
        assert_eq!(attractors[0].cell, 3);
        assert_eq!(attractors[1].overlaps, [1.0]);
        assert_eq!(attractors[2].nearest.overlap, -1.0);
        assert_eq!(attractors[1].energy, -0.5);
        assert_eq!(attractors[0].energy, 0.5);
        assert_eq!(attractors[0].mixture, None);
    }

    /// Two runs of the same count keep the order of their first runs.
    #[test]
    fn breaks_ties_by_the_first_run() {
        let runs = vec![run(2, 0, 0), run(0, 0, 1)];
        let attractors = Attractor::collect(&runs, &set(), &weights(), ZeroField::Keep);
        assert_eq!(attractors[0].class, FinalClass::Negation);
        assert_eq!(attractors[1].class, FinalClass::Pattern);
    }
}
