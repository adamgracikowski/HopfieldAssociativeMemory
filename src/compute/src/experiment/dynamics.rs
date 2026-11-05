//! Dynamics settings of an experiment.

use serde::{Deserialize, Serialize};

use crate::dynamics::{Dynamics, Mode, UpdateOrder, ZeroField};

/// The `[dynamics]` table of an experiment: the update modes to compare and
/// the settings shared by all of them, with the defaults of [`Dynamics`].
///
/// ```toml
/// [dynamics]
/// modes = ["synchronous", "asynchronous"]
/// zero_field = "keep"
/// max_steps = 1000
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExperimentDynamics {
    /// Update modes, one grid axis.
    pub modes: Vec<Mode>,
    /// Convention for `sgn(0)`; also decides pattern stability.
    pub zero_field: ZeroField,
    /// Order of updates within a sweep (asynchronous mode only).
    pub order: UpdateOrder,
    /// Largest number of steps (synchronous updates or asynchronous
    /// sweeps).
    pub max_steps: usize,
}

impl Default for ExperimentDynamics {
    /// The default mode and settings of [`Dynamics`].
    fn default() -> Self {
        let dynamics = Dynamics::default();
        Self {
            modes: vec![dynamics.mode],
            zero_field: dynamics.zero_field,
            order: dynamics.order,
            max_steps: dynamics.max_steps,
        }
    }
}

impl ExperimentDynamics {
    /// Settings of a single run in `mode`. Trials record no frames, so
    /// `record_every` keeps its default.
    pub fn for_mode(&self, mode: Mode) -> Dynamics {
        Dynamics {
            mode,
            zero_field: self.zero_field,
            order: self.order,
            max_steps: self.max_steps,
            ..Dynamics::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_in_the_defaults_of_dynamics() {
        let dynamics: ExperimentDynamics = toml::from_str("").unwrap();
        assert_eq!(dynamics.modes, [Mode::Asynchronous]);
        assert_eq!(dynamics.for_mode(Mode::Asynchronous), Dynamics::default());
    }

    #[test]
    fn builds_the_settings_of_each_mode() {
        let dynamics: ExperimentDynamics = toml::from_str(
            "modes = [\"synchronous\", \"asynchronous\"]\nzero_field = \"plus-one\"\n\
             order = \"sequential\"\nmax_steps = 20",
        )
        .unwrap();
        let synchronous = dynamics.for_mode(Mode::Synchronous);
        assert_eq!(
            synchronous,
            Dynamics {
                mode: Mode::Synchronous,
                zero_field: ZeroField::PlusOne,
                order: UpdateOrder::Sequential,
                max_steps: 20,
                record_every: None,
            }
        );
        assert!(toml::from_str::<ExperimentDynamics>("mode = \"synchronous\"").is_err());
        assert!(toml::from_str::<ExperimentDynamics>("record_every = 3").is_err());
    }
}
