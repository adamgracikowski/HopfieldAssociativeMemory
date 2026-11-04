//! Asynchronous dynamics: one neuron updated at a time.

use rand::Rng;

use super::{Outcome, Recorder, Termination, UpdateOrder, ZeroField};
use crate::patterns::{Pattern, PatternView};
use crate::weights::{FieldTracker, WeightMatrix, Weights};

/// Asynchronous simulation `x_i <- sgn(h_i)`, one neuron at a time, in
/// sweeps of `N` updates.
#[derive(Debug, Clone, Copy)]
pub(super) struct Asynchronous<'w> {
    pub(super) weights: &'w Weights,
    pub(super) zero_field: ZeroField,
    pub(super) order: UpdateOrder,
    pub(super) max_steps: usize,
    pub(super) record_every: usize,
}

impl Asynchronous<'_> {
    /// Runs from `initial` until a fixed point or `max_steps` sweeps.
    ///
    /// The fields come from a [`FieldTracker`], so an update costs `O(N)`
    /// for dense and `O(p)` for factored weights. A sweep without changes
    /// ends the run when every neuron was updated in it; for the `random`
    /// order, which may skip neurons, every neuron is checked first, and an
    /// unchanged sweep that is not a fixed point still counts as a step. At
    /// the step limit the state is checked for a fixed point as well.
    ///
    /// A frame is recorded every `record_every` updates if the state
    /// differs from the previous frame, and once more at the end if the
    /// final state has not been recorded yet. Its change count is the
    /// Hamming distance to the previous frame, which can be smaller than
    /// the number of flips in between when a neuron flips back (random
    /// order, asymmetric weights or frames longer than a sweep).
    pub(super) fn run<R: Rng + ?Sized>(
        &self,
        initial: PatternView<'_>,
        rng: &mut R,
        recorder: &mut impl Recorder,
    ) -> Outcome {
        let n = initial.len();
        let mut tracker = FieldTracker::new(self.weights, initial);
        let energy = self.weights.energy(initial.values());
        recorder.record(initial, energy, 0);

        let mut sweep = Vec::with_capacity(n);
        let mut steps = 0;
        let mut total = 0;
        let mut updates = 0;
        let mut frame = Frame {
            state: initial.to_pattern(),
            flips: 0,
        };
        let termination = loop {
            if steps == self.max_steps {
                break if self.is_fixed_point(&tracker) {
                    Termination::FixedPoint
                } else {
                    Termination::StepLimit
                };
            }
            self.order.fill_sweep(&mut sweep, n, rng);
            let mut changes = 0;
            for &i in &sweep {
                let x = tracker.state().values()[i];
                if self.zero_field.activate(tracker.field(i), x) != x {
                    tracker.flip(i);
                    changes += 1;
                    frame.flips += 1;
                }
                updates += 1;
                if updates % self.record_every == 0 {
                    frame.record_if_changed(tracker.state(), self.weights, recorder);
                }
            }
            total += changes;
            if changes == 0 && (self.order.covers_all_neurons() || self.is_fixed_point(&tracker)) {
                break Termination::FixedPoint;
            }
            steps += 1;
        };

        frame.record_if_changed(tracker.state(), self.weights, recorder);
        let state = tracker.into_state();
        let energy = self.weights.energy(state.values());
        Outcome {
            state,
            steps,
            changes: total,
            termination,
            energy,
        }
    }

    /// Whether no neuron would change if updated now.
    fn is_fixed_point(&self, tracker: &FieldTracker<'_>) -> bool {
        let state = tracker.state();
        (0..tracker.neurons()).all(|i| {
            let x = state.values()[i];
            self.zero_field.activate(tracker.field(i), x) == x
        })
    }
}

/// The last recorded frame and the number of flips since then.
struct Frame {
    state: Pattern,
    flips: usize,
}

impl Frame {
    /// Records `state` if it differs from the last frame, with its energy
    /// and the Hamming distance to the last frame. The distance is computed
    /// only after a flip, so frames without flips cost nothing.
    fn record_if_changed(
        &mut self,
        state: PatternView<'_>,
        weights: &Weights,
        recorder: &mut impl Recorder,
    ) {
        if self.flips == 0 {
            return;
        }
        self.flips = 0;
        let changes = state.hamming(&self.state);
        if changes > 0 {
            recorder.record(state, weights.energy(state.values()), changes);
            self.state = state.to_pattern();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Dynamics, History, Mode, NoHistory};
    use super::*;
    use crate::PatternSet;
    use crate::learning::{Hebb, LearningRule, Oja};
    use crate::patterns::Pattern;
    use crate::weights::{Backend, DenseWeights};
    use ndarray::{Array1, Array2, array};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use strum::IntoEnumIterator;

    fn asynchronous(order: UpdateOrder, max_steps: usize) -> Dynamics {
        Dynamics {
            mode: Mode::Asynchronous,
            order,
            max_steps,
            ..Dynamics::default()
        }
    }

    fn pattern(values: Array1<i8>) -> Pattern {
        Pattern::new(values).unwrap()
    }

    fn run(
        dynamics: Dynamics,
        weights: &Weights,
        initial: &Pattern,
        seed: u64,
    ) -> (Outcome, History) {
        let mut history = History::new();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let outcome = dynamics
            .run(weights, initial.view(), &mut rng, &mut history)
            .unwrap();
        (outcome, history)
    }

    fn random_set(n: usize, p: usize, rng: &mut ChaCha8Rng) -> PatternSet {
        let patterns: Vec<_> = (0..p).map(|_| Pattern::random(n, rng).unwrap()).collect();
        PatternSet::from_patterns(n, 1, &patterns).unwrap()
    }

    /// Hebbian weights of `(1, 1)`, `w_01 = w_10 = 0.5`, from `(1, -1)` in
    /// sequential order: neuron 0 has `h_0 = 0.5 * -1 < 0` and flips to
    /// `-1`; then neuron 1 has `h_1 = 0.5 * -1 < 0` and stays `-1`. The
    /// second sweep changes nothing, so `(-1, -1)`, the negated pattern, is
    /// a fixed point after one step with one change. Energy
    /// `-w_01 x_0 x_1` falls from `0.5` to `-0.5`.
    #[test]
    fn two_neurons_reach_the_negated_pattern() {
        let set = PatternSet::new(2, 1, array![[1, 1]]).unwrap();
        let weights = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let dynamics = asynchronous(UpdateOrder::Sequential, 100);
        let (outcome, history) = run(dynamics, &weights, &pattern(array![1, -1]), 0);
        assert_eq!(outcome.termination, Termination::FixedPoint);
        assert_eq!(outcome.state, pattern(array![-1, -1]));
        assert_eq!((outcome.steps, outcome.changes), (1, 1));
        assert_eq!(outcome.energy, -0.5);
        assert_eq!(history.states(), array![[1, -1], [-1, -1]]);
        assert_eq!(history.energies(), [0.5, -0.5]);
        assert_eq!(history.changes(), [0, 1]);
    }

    /// The asymmetric `W = [[0, -1], [1, 0]]` of Rojas (1996, p. 344) gives
    /// `h_0 = -x_1`, `h_1 = x_0`. In sequential order from `(1, 1)` every
    /// update flips a neuron: `(-1, 1)`, `(-1, -1)` in the first sweep,
    /// `(1, -1)`, `(1, 1)` in the second. The network cycles, so 2 sweeps
    /// end at the limit with 4 changes. A frame per update gives 1 + 4
    /// frames, a frame per sweep 1 + 2.
    #[test]
    fn asymmetric_weights_cycle_until_the_step_limit() {
        let weights = Weights::from(DenseWeights::new(array![[0.0, -1.0], [1.0, 0.0]]).unwrap());
        let start = pattern(array![1, 1]);
        let per_update = Dynamics {
            record_every: Some(1),
            ..asynchronous(UpdateOrder::Sequential, 2)
        };
        let (outcome, history) = run(per_update, &weights, &start, 0);
        assert_eq!(outcome.termination, Termination::StepLimit);
        assert_eq!(outcome.state, start);
        assert_eq!((outcome.steps, outcome.changes), (2, 4));
        assert_eq!(
            history.states(),
            array![[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]
        );
        assert_eq!(history.changes(), [0, 1, 1, 1, 1]);

        let (_, history) = run(
            asynchronous(UpdateOrder::Sequential, 2),
            &weights,
            &start,
            0,
        );
        assert_eq!(history.states(), array![[1, 1], [-1, -1], [1, 1]]);
        assert_eq!(history.changes(), [0, 2, 2]);
    }

    /// The same cycle with a frame every 4 updates: after two sweeps the
    /// state is back at `(1, 1)` after 4 flips, so no new frame is recorded.
    /// With a frame every 3 updates the frame after 3 flips is `(1, -1)`,
    /// at distance 1 from the start, and the final `(1, 1)` follows at
    /// distance 1.
    #[test]
    fn frames_record_the_distance_to_the_previous_frame() {
        let weights = Weights::from(DenseWeights::new(array![[0.0, -1.0], [1.0, 0.0]]).unwrap());
        let start = pattern(array![1, 1]);
        let every = |k| Dynamics {
            record_every: Some(k),
            ..asynchronous(UpdateOrder::Sequential, 2)
        };
        let (outcome, history) = run(every(4), &weights, &start, 0);
        assert_eq!(outcome.changes, 4);
        assert_eq!(history.states(), array![[1, 1]]);

        let (_, history) = run(every(3), &weights, &start, 0);
        assert_eq!(history.states(), array![[1, 1], [1, -1], [1, 1]]);
        assert_eq!(history.changes(), [0, 1, 1]);
    }

    /// With the random order a neuron can flip back within a frame, so the
    /// recorded changes are compared with the actual distances.
    #[test]
    fn random_order_frames_are_distinct_and_counted_exactly() {
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let set = random_set(40, 8, &mut rng);
        let weights = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let dynamics = asynchronous(UpdateOrder::Random, 1000);
        for seed in 0..50 {
            let start = Pattern::random(40, &mut rng).unwrap();
            let (_, history) = run(dynamics, &weights, &start, seed);
            for k in 1..history.len() {
                let distance = history.state(k).hamming(&history.state(k - 1));
                assert!(distance > 0);
                assert_eq!(history.changes()[k], distance, "{seed} {k}");
            }
        }
    }

    /// The two-neuron Hebbian network from `(1, -1)` reaches `(-1, -1)` in
    /// the first sweep; with a limit of one step the check at the limit
    /// finds a fixed point.
    #[test]
    fn a_fixed_point_at_the_step_limit_is_recognized() {
        let set = PatternSet::new(2, 1, array![[1, 1]]).unwrap();
        let weights = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let (outcome, _) = run(
            asynchronous(UpdateOrder::Sequential, 1),
            &weights,
            &pattern(array![1, -1]),
            0,
        );
        assert_eq!(outcome.termination, Termination::FixedPoint);
        assert_eq!(outcome.steps, 1);
    }

    /// Symmetric weights with zero diagonal: the energy never increases
    /// between updates and the run ends in a fixed point.
    #[test]
    fn energy_never_increases_with_symmetric_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let set = random_set(40, 6, &mut rng);
        for backend in [Backend::Dense, Backend::Factored] {
            let networks = [
                Hebb.train(&set, backend).unwrap().weights,
                Oja::default().train(&set, backend).unwrap().weights,
            ];
            for weights in &networks {
                for order in UpdateOrder::iter() {
                    let dynamics = Dynamics {
                        record_every: Some(1),
                        ..asynchronous(order, 1000)
                    };
                    for seed in 0..5 {
                        let start = Pattern::random(40, &mut rng).unwrap();
                        let (outcome, history) = run(dynamics, weights, &start, seed);
                        assert_eq!(outcome.termination, Termination::FixedPoint);
                        for pair in history.energies().windows(2) {
                            assert!(pair[1] <= pair[0] + 1e-12, "{order} {pair:?}");
                        }
                        assert_eq!(*history.energies().last().unwrap(), outcome.energy);
                    }
                }
            }
        }
    }

    /// With `W = 0`, `plus-one` and a single `-1` at neuron 7, only that
    /// neuron is unstable. A random sweep of 30 draws misses it with
    /// probability `(29/30)^30`, about 0.36; such a sweep changes nothing
    /// but is no fixed point, so it counts as a step and the run goes on.
    #[test]
    fn random_order_confirms_a_fixed_point() {
        let weights = Weights::from(DenseWeights::zeros(30));
        let start = pattern(Array1::from_shape_fn(30, |i| if i == 7 { -1 } else { 1 }));
        let dynamics = Dynamics {
            zero_field: ZeroField::PlusOne,
            ..asynchronous(UpdateOrder::Random, 100)
        };
        let mut missed = false;
        for seed in 0..20 {
            let (outcome, _) = run(dynamics, &weights, &start, seed);
            assert_eq!(outcome.termination, Termination::FixedPoint);
            assert_eq!(outcome.state, pattern(Array1::ones(30)));
            assert_eq!(outcome.changes, 1);
            missed |= outcome.steps > 1;
        }
        assert!(missed);
    }

    /// Direct asynchronous simulation on the full matrix with the same
    /// sweeps and stopping rules: final state, steps, changes and whether
    /// it is a fixed point.
    fn reference(
        w: &Array2<f64>,
        initial: &Array1<i8>,
        order: UpdateOrder,
        max_steps: usize,
        rng: &mut ChaCha8Rng,
    ) -> (Array1<i8>, usize, usize, bool) {
        let n = initial.len();
        let field =
            |x: &Array1<i8>, i: usize| -> f64 { (0..n).map(|j| w[[i, j]] * f64::from(x[j])).sum() };
        let stable =
            |x: &Array1<i8>| (0..n).all(|i| ZeroField::Keep.activate(field(x, i), x[i]) == x[i]);
        let mut x = initial.clone();
        let (mut steps, mut total) = (0, 0);
        let mut sweep = Vec::new();
        while steps < max_steps {
            order.fill_sweep(&mut sweep, n, rng);
            let mut changes = 0;
            for &i in &sweep {
                let new = ZeroField::Keep.activate(field(&x, i), x[i]);
                if new != x[i] {
                    x[i] = new;
                    changes += 1;
                }
            }
            total += changes;
            if changes == 0 && stable(&x) {
                return (x, steps, total, true);
            }
            steps += 1;
        }
        let fixed = stable(&x);
        (x, steps, total, fixed)
    }

    #[test]
    fn matches_a_direct_simulation_with_both_backends() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let set = random_set(30, 5, &mut rng);
        let dense = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let factored = Hebb.train(&set, Backend::Factored).unwrap().weights;
        let w = dense.to_dense();
        for order in UpdateOrder::iter() {
            for seed in 0..10 {
                let start = Pattern::random(30, &mut rng).unwrap();
                let mut reference_rng = ChaCha8Rng::seed_from_u64(seed);
                let (state, steps, changes, fixed) = reference(
                    &w,
                    &start.clone().into_inner(),
                    order,
                    50,
                    &mut reference_rng,
                );
                let (outcome, history) = run(asynchronous(order, 50), &dense, &start, seed);
                assert_eq!(outcome.state.values(), state.view(), "{order} {seed}");
                assert_eq!((outcome.steps, outcome.changes), (steps, changes));
                assert_eq!(outcome.termination == Termination::FixedPoint, fixed);

                let (other, other_history) = run(asynchronous(order, 50), &factored, &start, seed);
                assert_eq!(other.state, outcome.state);
                assert_eq!(
                    (other.steps, other.termination),
                    (outcome.steps, outcome.termination)
                );
                assert_eq!(other_history.states(), history.states());
            }
        }
    }

    #[test]
    fn same_seed_gives_the_same_run() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let set = random_set(50, 8, &mut rng);
        let weights = Hebb.train(&set, Backend::Dense).unwrap().weights;
        let start = Pattern::random(50, &mut rng).unwrap();
        for order in UpdateOrder::iter() {
            let dynamics = asynchronous(order, 100);
            assert_eq!(
                run(dynamics, &weights, &start, 9),
                run(dynamics, &weights, &start, 9)
            );
        }
    }

    #[test]
    fn records_nothing_without_history() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let set = random_set(20, 3, &mut rng);
        let weights = Hebb.train(&set, Backend::Factored).unwrap().weights;
        let start = Pattern::random(20, &mut rng).unwrap();
        let dynamics = Dynamics::default();
        let (expected, _) = run(dynamics, &weights, &start, 5);
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let outcome = dynamics
            .run(&weights, start.view(), &mut rng, &mut NoHistory)
            .unwrap();
        assert_eq!(outcome, expected);
    }
}
