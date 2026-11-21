//! Reasons for which a simulation stops.

use serde::Serialize;

/// Why a simulation stopped.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    strum::Display,
    strum::IntoStaticStr,
    strum::EnumIter,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum Termination {
    /// The state is a fixed point: updating any neuron leaves it unchanged.
    FixedPoint,
    /// Synchronous mode only: the next state equals the previous one.
    TwoCycle,
    /// The step limit was reached.
    StepLimit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_kebab_case() {
        assert_eq!(Termination::FixedPoint.to_string(), "fixed-point");
        assert_eq!(Termination::TwoCycle.to_string(), "two-cycle");
        let name: &str = Termination::StepLimit.into();
        assert_eq!(name, "step-limit");
    }
}
