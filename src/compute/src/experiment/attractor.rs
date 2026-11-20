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
/// ended in it; a 2-cycle is represented by its lexicographically smaller
/// state.
#[derive(Debug, Clone, PartialEq)]
pub struct Attractor {
    /// Index of the grid cell.
    pub cell: usize,
    /// Position in the cell by decreasing `count`, then by first run.
    pub rank: usize,
    /// Number of runs that ended in this state.
    pub count: usize,
    /// Fraction `count / runs` of the runs of the cell.
    pub fraction: f64,
    /// Class of the final state.
    pub class: FinalClass,
    /// Why the runs stopped.
    pub termination: Termination,
    /// Energy of the state, computed from the weights.
    pub energy: f64,
    /// Overlaps `m` of the state with every stored pattern.
    pub overlaps: Vec<f64>,
    /// Stored pattern with the largest `|m|`.
    pub nearest: Nearest,
    /// The [closest](Mixture::closest) mixture of stored patterns.
    pub mixture: Option<Mixture>,
    /// The first run that ended in this state; `recall` repeats it.
    pub first: Trial,
    /// The state.
    pub state: Pattern,
}

impl Attractor {
    /// The distinct final states of the runs of one cell, grouped by
    /// termination and state; `zero_field` gives the second state of a
    /// 2-cycle.
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
        let mut groups: BTreeMap<(&'static str, Vec<i8>), (usize, usize)> = BTreeMap::new();
        for (index, (result, state)) in runs.iter().enumerate() {
            let state = Self::canonical(state, result.termination, weights, zero_field);
            let key = (result.termination.into(), state.values().to_vec());
            groups
                .entry(key)
                .and_modify(|(count, _)| *count += 1)
                .or_insert((1, index));
        }
        let mut groups: Vec<(usize, usize, Pattern)> = groups
            .into_iter()
            .map(|((_, values), (count, first))| {
                (count, first, Pattern::new_unchecked(Array1::from(values)))
            })
            .collect();
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

    /// `state`, or for a 2-cycle the smaller of it and its successor.
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

    /// Hebbian weights of `(1, 1)`: `w_01 = w_10 = 0.5`.
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

    /// `(1, 1)` and its negation `(-1, -1)` are fixed points; one flipped
    /// pixel gives the 2-cycle of `(1, -1)` and `(-1, 1)`, represented by the
    /// smaller `(-1, 1)` whichever state it ends in.
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

    #[test]
    fn breaks_ties_by_the_first_run() {
        let runs = vec![run(2, 0, 0), run(0, 0, 1)];
        let attractors = Attractor::collect(&runs, &set(), &weights(), ZeroField::Keep);
        assert_eq!(attractors[0].class, FinalClass::Negation);
        assert_eq!(attractors[1].class, FinalClass::Pattern);
    }
}
