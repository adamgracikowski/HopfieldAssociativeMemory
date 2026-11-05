//! Classes of the final state of a trial.

use serde::Serialize;

use crate::PatternSet;
use crate::dynamics::{Outcome, Termination};

/// What a trial ended in, decided by exact equality of states.
///
/// [`Display`](std::fmt::Display) and `&'static str` conversion give the
/// name used in output files (`pattern`, `negation`, `spurious`, `cycle`,
/// `unconverged`).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    strum::Display,
    strum::IntoStaticStr,
    strum::EnumIter,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum FinalClass {
    /// A fixed point equal to a stored pattern.
    Pattern,
    /// A fixed point equal to a negated stored pattern.
    Negation,
    /// Any other fixed point, e.g. a mixture of patterns.
    Spurious,
    /// A cycle of length 2 (synchronous mode).
    Cycle,
    /// The step limit was reached.
    Unconverged,
}

impl FinalClass {
    /// Class of `outcome` in a network that stores `set`. A stored pattern
    /// has Hamming distance 0 to the state, a negated one distance `N`.
    pub fn of(outcome: &Outcome, set: &PatternSet) -> Self {
        match outcome.termination {
            Termination::TwoCycle => Self::Cycle,
            Termination::StepLimit => Self::Unconverged,
            Termination::FixedPoint => {
                let state = &outcome.state;
                let distances: Vec<usize> = set.iter().map(|xi| state.hamming(&xi)).collect();
                if distances.contains(&0) {
                    Self::Pattern
                } else if distances.contains(&set.neurons()) {
                    Self::Negation
                } else {
                    Self::Spurious
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::Pattern;
    use ndarray::{Array1, array};
    use strum::IntoEnumIterator;

    fn set() -> PatternSet {
        PatternSet::new(4, 1, array![[1, 1, 1, 1], [1, -1, 1, -1]]).unwrap()
    }

    fn class(state: Array1<i8>, termination: Termination) -> FinalClass {
        let outcome = Outcome {
            state: Pattern::new(state).unwrap(),
            steps: 1,
            changes: 1,
            termination,
            energy: 0.0,
        };
        FinalClass::of(&outcome, &set())
    }

    /// `(-1, 1, -1, 1)` is the negation of the second pattern; `(1, 1, -1,
    /// -1)` has distance 2 to both patterns and their negations.
    #[test]
    fn classifies_by_exact_equality() {
        let fixed = Termination::FixedPoint;
        assert_eq!(class(array![1, -1, 1, -1], fixed), FinalClass::Pattern);
        assert_eq!(class(array![-1, 1, -1, 1], fixed), FinalClass::Negation);
        assert_eq!(class(array![1, 1, -1, -1], fixed), FinalClass::Spurious);
        assert_eq!(
            class(array![1, -1, 1, -1], Termination::TwoCycle),
            FinalClass::Cycle
        );
        assert_eq!(
            class(array![1, 1, 1, 1], Termination::StepLimit),
            FinalClass::Unconverged
        );
    }

    #[test]
    fn names_the_classes() {
        let names: Vec<&str> = FinalClass::iter().map(Into::into).collect();
        assert_eq!(
            names,
            ["pattern", "negation", "spurious", "cycle", "unconverged"]
        );
        assert_eq!(
            serde_json::to_value(FinalClass::Unconverged).unwrap(),
            "unconverged"
        );
    }
}
