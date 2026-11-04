//! Recorded frames of a simulation.

use ndarray::{ArrayView1, ArrayView2};

use super::Recorder;
use crate::patterns::{BipolarVector, PatternView};

/// Every recorded frame of a simulation: the states (one row per frame),
/// their energies and the number of neurons changed since the previous
/// frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    neurons: usize,
    states: Vec<i8>,
    energies: Vec<f64>,
    changes: Vec<usize>,
}

impl History {
    /// Empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Whether no frame has been recorded.
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }

    /// States as a `frames x N` matrix.
    pub fn states(&self) -> ArrayView2<'_, i8> {
        ArrayView2::from_shape((self.len(), self.neurons), &self.states)
            .expect("every frame has `neurons` values")
    }

    /// State of frame `k`.
    ///
    /// # Panics
    ///
    /// Panics if `k >= self.len()`.
    pub fn state(&self, k: usize) -> PatternView<'_> {
        assert!(k < self.len(), "frame {k} of {}", self.len());
        let start = k * self.neurons;
        BipolarVector::new_unchecked(ArrayView1::from(&self.states[start..start + self.neurons]))
    }

    /// Energy of every frame.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Number of neurons changed since the previous frame, for every frame.
    pub fn changes(&self) -> &[usize] {
        &self.changes
    }
}

impl Recorder for History {
    /// # Panics
    ///
    /// Panics if `state` has a different length than the earlier frames.
    fn record(&mut self, state: PatternView<'_>, energy: f64, changes: usize) {
        if self.is_empty() {
            self.neurons = state.len();
        }
        assert_eq!(state.len(), self.neurons, "frame length");
        self.states.extend(state.values().iter());
        self.energies.push(energy);
        self.changes.push(changes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::Pattern;
    use ndarray::array;

    #[test]
    fn keeps_frames_in_order() {
        let a = Pattern::new(array![1, -1, 1]).unwrap();
        let b = Pattern::new(array![1, 1, 1]).unwrap();
        let mut history = History::new();
        assert!(history.is_empty());
        history.record(a.view(), 0.5, 0);
        history.record(b.view(), -1.5, 1);
        assert_eq!(history.len(), 2);
        assert_eq!(history.states(), array![[1, -1, 1], [1, 1, 1]]);
        assert_eq!(history.state(1), b);
        assert_eq!(history.energies(), [0.5, -1.5]);
        assert_eq!(history.changes(), [0, 1]);
    }

    #[test]
    #[should_panic(expected = "frame length")]
    fn rejects_frames_of_another_length() {
        let mut history = History::new();
        history.record(Pattern::new(array![1, -1]).unwrap().view(), 0.0, 0);
        history.record(Pattern::new(array![1]).unwrap().view(), 0.0, 0);
    }
}
