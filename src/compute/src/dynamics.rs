//! Recall dynamics of a Hopfield network.
//!
//! [`Dynamics`] runs a network from an initial state in one of the update
//! [`Mode`]s until it reaches a fixed point, a cycle of length 2 or the
//! step limit, and returns the [`Outcome`]. A [`Recorder`] receives every
//! frame on the way: [`History`] keeps them, [`NoHistory`] drops them.

mod asynchronous;
pub mod history;
pub mod mode;
pub mod order;
pub mod outcome;
pub mod recorder;
pub mod stability;
mod synchronous;
pub mod termination;
pub mod zero_field;

use rand::Rng;
use serde::{Deserialize, Serialize};

pub use history::History;
pub use mode::Mode;
pub use order::UpdateOrder;
pub use outcome::Outcome;
pub use recorder::{NoHistory, Recorder};
pub use stability::Stability;
pub use termination::Termination;
pub use zero_field::{ZERO_FIELD_TOLERANCE, ZeroField};

use crate::patterns::PatternView;
use crate::weights::{WeightMatrix, Weights};
use crate::{Error, Result};
use asynchronous::Asynchronous;
use synchronous::Synchronous;

/// Settings of a recall simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Dynamics {
    /// Update mode.
    pub mode: Mode,
    /// Convention for `sgn(0)`.
    pub zero_field: ZeroField,
    /// Order of updates within a sweep (asynchronous mode only).
    pub order: UpdateOrder,
    /// Largest number of steps (synchronous updates or asynchronous
    /// sweeps).
    pub max_steps: usize,
    /// Asynchronous mode only: record a frame every `record_every` updates;
    /// `None` means once per sweep.
    pub record_every: Option<usize>,
}

impl Default for Dynamics {
    fn default() -> Self {
        Self {
            mode: Mode::Asynchronous,
            zero_field: ZeroField::Keep,
            order: UpdateOrder::Permutation,
            max_steps: 1000,
            record_every: None,
        }
    }
}

impl Dynamics {
    /// Runs the network from `initial`, passing the frames to `recorder`;
    /// `rng` drives the random update orders of the asynchronous mode.
    pub fn run<R, H>(
        &self,
        weights: &Weights,
        initial: PatternView<'_>,
        rng: &mut R,
        recorder: &mut H,
    ) -> Result<Outcome>
    where
        R: Rng + ?Sized,
        H: Recorder,
    {
        self.validate()?;
        if initial.len() != weights.neurons() {
            return Err(Error::InvalidArgument {
                message: format!(
                    "initial state has {} neurons, the network has {}",
                    initial.len(),
                    weights.neurons()
                ),
            });
        }
        match self.mode {
            Mode::Synchronous => Ok(Synchronous {
                weights,
                zero_field: self.zero_field,
                max_steps: self.max_steps,
            }
            .run(initial, recorder)),
            Mode::Asynchronous => Ok(Asynchronous {
                weights,
                zero_field: self.zero_field,
                order: self.order,
                max_steps: self.max_steps,
                record_every: self.record_every.unwrap_or(weights.neurons()),
            }
            .run(initial, rng, recorder)),
        }
    }

    fn validate(&self) -> Result<()> {
        let invalid = |message: &str| {
            Err(Error::InvalidArgument {
                message: message.to_owned(),
            })
        };
        if self.max_steps == 0 {
            return invalid("at least one step is needed");
        }
        if self.record_every == Some(0) {
            return invalid("frames must be recorded at least every update, not every 0");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PatternSet;
    use crate::learning::{Hebb, LearningRule};
    use crate::patterns::Pattern;
    use crate::weights::{Backend, DenseWeights};
    use ndarray::{Array1, Array2, array};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn synchronous(max_steps: usize) -> Dynamics {
        Dynamics {
            mode: Mode::Synchronous,
            max_steps,
            ..Dynamics::default()
        }
    }

    fn pattern(values: Array1<i8>) -> Pattern {
        Pattern::new(values).unwrap()
    }

    fn run(dynamics: Dynamics, weights: &Weights, initial: &Pattern) -> (Outcome, History) {
        let mut history = History::new();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let outcome = dynamics
            .run(weights, initial.view(), &mut rng, &mut history)
            .unwrap();
        (outcome, history)
    }

    /// Hebbian weights of the single pattern `(1, 1)`:
    /// `w_01 = w_10 = 1 * 1 / 2 = 0.5`.
    fn two_neurons() -> Weights {
        let set = PatternSet::new(2, 1, array![[1, 1]]).unwrap();
        Hebb.train(&set, Backend::Dense).unwrap().weights
    }

    /// From `(1, -1)`: `h = (0.5 * -1, 0.5 * 1) = (-0.5, 0.5)`, so
    /// `x(1) = (-1, 1)`; then `h = (0.5, -0.5)`, so `x(2) = (1, -1) = x(0)`:
    /// a cycle of length 2 found after one applied step with 2 changes.
    /// Both states have energy `-w_01 x_0 x_1 = 0.5`.
    #[test]
    fn two_neurons_oscillate_in_synchronous_mode() {
        let (outcome, history) = run(synchronous(100), &two_neurons(), &pattern(array![1, -1]));
        assert_eq!(outcome.termination, Termination::TwoCycle);
        assert_eq!(outcome.state, pattern(array![-1, 1]));
        assert_eq!((outcome.steps, outcome.changes), (1, 2));
        assert_eq!(outcome.energy, 0.5);
        assert_eq!(history.states(), array![[1, -1], [-1, 1]]);
        assert_eq!(history.energies(), [0.5, 0.5]);
        assert_eq!(history.changes(), [0, 2]);
    }

    /// From `(1, -1)` the 2-cycle appears after one step, so a limit of one
    /// step reports it rather than the step limit.
    #[test]
    fn a_two_cycle_at_the_step_limit_is_recognized() {
        let (outcome, _) = run(synchronous(1), &two_neurons(), &pattern(array![1, -1]));
        assert_eq!(outcome.termination, Termination::TwoCycle);
        assert_eq!(outcome.steps, 1);
    }

    /// `h = (0.5, 0.5)` keeps `(1, 1)`, with energy `-0.5`.
    #[test]
    fn stored_pattern_is_a_fixed_point() {
        let stored = pattern(array![1, 1]);
        let (outcome, history) = run(synchronous(100), &two_neurons(), &stored);
        assert_eq!(outcome.termination, Termination::FixedPoint);
        assert_eq!(outcome.state, stored);
        assert_eq!((outcome.steps, outcome.changes), (0, 0));
        assert_eq!(outcome.energy, -0.5);
        assert_eq!(history.len(), 1);
    }

    /// With `W = 0` every field is zero: `keep` leaves the state as it is,
    /// `plus-one` sets the single `-1` to `+1` in one step.
    #[test]
    fn zero_fields_follow_the_convention() {
        let weights = Weights::from(DenseWeights::zeros(3));
        let initial = pattern(array![1, -1, 1]);
        let (outcome, _) = run(synchronous(10), &weights, &initial);
        assert_eq!(
            (outcome.termination, outcome.steps),
            (Termination::FixedPoint, 0)
        );
        assert_eq!(outcome.state, initial);

        let plus_one = Dynamics {
            zero_field: ZeroField::PlusOne,
            ..synchronous(10)
        };
        let (outcome, _) = run(plus_one, &weights, &initial);
        assert_eq!(
            (outcome.termination, outcome.steps),
            (Termination::FixedPoint, 1)
        );
        assert_eq!(outcome.state, pattern(array![1, 1, 1]));
    }

    /// `W = [[0, -1], [1, 0]]` gives `h = (-x_1, x_0)`: a 4-cycle
    /// `(1, 1) -> (-1, 1) -> (-1, -1) -> (1, -1)` with one change per step,
    /// so `x(10) = x(2) = (-1, -1)`.
    #[test]
    fn longer_cycles_end_at_the_step_limit() {
        let weights = Weights::from(DenseWeights::new(array![[0.0, -1.0], [1.0, 0.0]]).unwrap());
        let (outcome, history) = run(synchronous(10), &weights, &pattern(array![1, 1]));
        assert_eq!(outcome.termination, Termination::StepLimit);
        assert_eq!(outcome.state, pattern(array![-1, -1]));
        assert_eq!((outcome.steps, outcome.changes), (10, 10));
        assert_eq!(history.len(), 11);
    }

    /// Direct synchronous simulation on the full matrix: the states until
    /// the first repeated state of distance 1 or 2, or `max_steps` steps,
    /// and the reason for stopping.
    fn reference(
        w: &Array2<f64>,
        initial: &Array1<i8>,
        max_steps: usize,
    ) -> (Vec<Array1<i8>>, Termination) {
        let mut states = vec![initial.clone()];
        loop {
            let x = states.last().unwrap();
            let next: Array1<i8> = (0..x.len())
                .map(|i| {
                    let h: f64 = (0..x.len()).map(|j| w[[i, j]] * f64::from(x[j])).sum();
                    ZeroField::Keep.activate(h, x[i])
                })
                .collect();
            let len = states.len();
            if next == states[len - 1] {
                return (states, Termination::FixedPoint);
            }
            if len >= 2 && next == states[len - 2] {
                return (states, Termination::TwoCycle);
            }
            if len - 1 == max_steps {
                return (states, Termination::StepLimit);
            }
            states.push(next);
        }
    }

    #[test]
    fn matches_a_direct_simulation_with_both_backends() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let patterns: Vec<_> = (0..6)
            .map(|_| Pattern::random(30, &mut rng).unwrap())
            .collect();
        let set = PatternSet::from_patterns(30, 1, &patterns).unwrap();
        let dense = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let factored = Hebb.train(&set, Backend::Factored).unwrap().weights;
        let w = dense.to_dense();
        for _ in 0..20 {
            let initial = Pattern::random(30, &mut rng).unwrap();
            let (expected, termination) = reference(&w, &initial.clone().into_inner(), 50);
            let (outcome, history) = run(synchronous(50), &dense, &initial);
            assert_eq!(outcome.termination, termination);
            assert_eq!(history.len(), expected.len());
            for (k, state) in expected.iter().enumerate() {
                assert_eq!(history.state(k).values(), state.view(), "frame {k}");
            }
            assert_eq!(outcome.steps, expected.len() - 1);
            let (other, other_history) = run(synchronous(50), &factored, &initial);
            assert_eq!(other.state, outcome.state);
            assert_eq!(other.termination, outcome.termination);
            assert_eq!(other_history.states(), history.states());
            assert!((other.energy - outcome.energy).abs() < 1e-12);
        }
    }

    #[test]
    fn records_nothing_without_history() {
        let weights = two_neurons();
        let initial = pattern(array![1, -1]);
        let (expected, _) = run(synchronous(100), &weights, &initial);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let outcome = synchronous(100)
            .run(&weights, initial.view(), &mut rng, &mut NoHistory)
            .unwrap();
        assert_eq!(outcome, expected);
    }

    #[test]
    fn rejects_invalid_settings_and_states() {
        let weights = two_neurons();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let short = pattern(array![1]);
        let error = synchronous(10)
            .run(&weights, short.view(), &mut rng, &mut NoHistory)
            .unwrap_err();
        assert!(error.to_string().contains("has 1 neurons"), "{error}");
        let initial = pattern(array![1, 1]);
        let error = synchronous(0)
            .run(&weights, initial.view(), &mut rng, &mut NoHistory)
            .unwrap_err();
        assert!(error.to_string().contains("at least one step"), "{error}");
        let never = Dynamics {
            record_every: Some(0),
            ..Dynamics::default()
        };
        let error = never
            .run(&weights, initial.view(), &mut rng, &mut NoHistory)
            .unwrap_err();
        assert!(error.to_string().contains("every 0"), "{error}");
    }

    #[test]
    fn parses_settings() {
        let dynamics: Dynamics =
            toml::from_str("mode = \"synchronous\"\nzero_field = \"plus-one\"\nmax_steps = 20")
                .unwrap();
        assert_eq!(
            dynamics,
            Dynamics {
                mode: Mode::Synchronous,
                zero_field: ZeroField::PlusOne,
                max_steps: 20,
                ..Dynamics::default()
            }
        );
        let dynamics: Dynamics =
            toml::from_str("order = \"sequential\"\nrecord_every = 5").unwrap();
        assert_eq!(dynamics.order, UpdateOrder::Sequential);
        assert_eq!(dynamics.record_every, Some(5));
        assert_eq!(toml::from_str::<Dynamics>("").unwrap(), Dynamics::default());
        assert!(toml::from_str::<Dynamics>("steps = 3").is_err());
    }
}
