//! Reasons for which a simulation stops.

/// Why a simulation stopped.
///
/// [`Display`](std::fmt::Display) and `&'static str` conversion give the
/// name used in output files (`fixed-point`, `two-cycle`, `step-limit`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, strum::Display, strum::IntoStaticStr, strum::EnumIter,
)]
#[strum(serialize_all = "kebab-case")]
pub enum Termination {
    /// The state is a fixed point: updating any neuron leaves it unchanged.
    FixedPoint,
    /// Synchronous mode only: the next state equals the previous one, so
    /// the network alternates between two states.
    TwoCycle,
    /// The step limit was reached without a fixed point or a cycle of
    /// length 2.
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
