//! Update modes of the network.

use serde::Deserialize;

/// How neurons are updated in one step.
///
/// [`Display`](std::fmt::Display) and `&'static str` conversion give the
/// name used in configuration files (`synchronous`, `asynchronous`).
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Deserialize,
    strum::Display,
    strum::IntoStaticStr,
    strum::EnumIter,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum Mode {
    /// All neurons at once from the same state: `x(t+1) = sgn(W x(t))`.
    /// With symmetric weights the network ends in a fixed point or a cycle
    /// of length 2 (Goles et al., after Bruck 1990, Theorem 4.2).
    Synchronous,
    /// One neuron at a time, each from the current state: `x_i <- sgn(h_i)`.
    /// With symmetric weights and zero diagonal the energy never increases,
    /// so the network ends in a fixed point (Hopfield 1982). A step is one
    /// sweep of `N` updates.
    #[default]
    Asynchronous,
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    #[test]
    fn names_round_trip_through_configuration() {
        #[derive(Deserialize)]
        struct Wrapper {
            mode: Mode,
        }
        for mode in Mode::iter() {
            let text = format!("mode = \"{mode}\"");
            assert_eq!(toml::from_str::<Wrapper>(&text).unwrap().mode, mode);
        }
        assert_eq!(Mode::Synchronous.to_string(), "synchronous");
        assert_eq!(Mode::default(), Mode::Asynchronous);
    }
}
