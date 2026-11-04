//! Synchronous dynamics: all neurons updated at once.

use ndarray::{Array1, Zip};

use super::{Outcome, Recorder, Termination, ZeroField};
use crate::patterns::{BipolarVector, PatternView};
use crate::weights::{WeightMatrix, Weights, energy_from_fields};

/// Synchronous simulation `x(t+1) = sgn(W x(t))`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Synchronous<'w> {
    pub(super) weights: &'w Weights,
    pub(super) zero_field: ZeroField,
    pub(super) max_steps: usize,
}

impl Synchronous<'_> {
    /// Runs from `initial` until a fixed point, a cycle of length 2 or
    /// `max_steps` applied steps.
    ///
    /// Every step computes the fields `h = W x(t)` (in parallel for large
    /// networks), records `x(t)` with its energy `-1/2 x(t)^T h` and builds
    /// the next state. Without changes `x(t)` is a fixed point; if the next
    /// state equals `x(t-1)`, the network alternates between `x(t-1)` and
    /// `x(t)`. Only three state vectors and one field vector are kept.
    pub(super) fn run(&self, initial: PatternView<'_>, recorder: &mut impl Recorder) -> Outcome {
        let n = initial.len();
        let mut current = initial.values().to_owned();
        let mut next = Array1::zeros(n);
        let mut previous: Option<Array1<i8>> = None;
        let mut fields = Array1::zeros(n);
        let mut changes = 0;
        let mut total = 0;
        let mut t = 0;
        let termination = loop {
            self.weights
                .local_fields_into(current.view(), fields.view_mut());
            let energy = energy_from_fields(current.view(), fields.view());
            recorder.record(
                BipolarVector::new_unchecked(current.view()),
                energy,
                changes,
            );

            Zip::from(&mut next)
                .and(&current)
                .and(&fields)
                .for_each(|x, &old, &h| *x = self.zero_field.activate(h, old));
            changes = next.iter().zip(&current).filter(|(a, b)| a != b).count();
            if changes == 0 {
                break Termination::FixedPoint;
            }
            if previous.as_ref() == Some(&next) {
                break Termination::TwoCycle;
            }
            if t == self.max_steps {
                break Termination::StepLimit;
            }
            total += changes;
            t += 1;
            std::mem::swap(&mut current, &mut next);
            match previous.as_mut() {
                Some(buffer) => std::mem::swap(buffer, &mut next),
                None => previous = Some(std::mem::replace(&mut next, Array1::zeros(n))),
            }
        };
        let energy = energy_from_fields(current.view(), fields.view());
        Outcome {
            state: BipolarVector::new_unchecked(current),
            steps: t,
            changes: total,
            termination,
            energy,
        }
    }
}
