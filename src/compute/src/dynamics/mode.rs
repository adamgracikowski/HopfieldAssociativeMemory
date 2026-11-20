//! Update modes of the network.

use serde::{Deserialize, Serialize};

/// How neurons are updated in one step.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    strum::Display,
    strum::IntoStaticStr,
    strum::EnumIter,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum Mode {
    /// `x(t+1) = sgn(W x(t))`; ends in a fixed point or a 2-cycle (Goles et
    /// al., after Bruck 1990, Theorem 4.2).
    Synchronous,
    /// One neuron at a time, a step is a sweep of `N` updates; ends in a
    /// fixed point (Hopfield 1982).
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
