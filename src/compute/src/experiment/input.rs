//! Initial states of the trials of an experiment.

use serde::{Deserialize, Serialize};

/// The `[input]` table of an experiment, whose `source` key names the kind
/// of initial state:
///
/// ```toml
/// [input]
/// source = "pattern"           # every stored pattern with flipped pixels
/// noise = [0.0, 0.1, 0.2]      # fractions of flipped pixels, one grid axis
/// ```
///
/// `source = "random"` takes no parameters: every trial starts from one
/// random state.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ExperimentInput {
    /// In every trial each stored pattern in turn, with
    /// [`flips`](Self::flips) random pixels flipped.
    Pattern {
        /// Fractions of flipped pixels, from 0 to 1.
        noise: Vec<f64>,
    },
    /// In every trial one random state, each pixel `1` or `-1` with
    /// probability 1/2.
    Random {},
}

impl ExperimentInput {
    /// Name of the source as written in the configuration.
    pub fn source(&self) -> &'static str {
        match self {
            Self::Pattern { .. } => "pattern",
            Self::Random {} => "random",
        }
    }

    /// Number of pixels flipped at noise level `noise` in a network of
    /// `neurons` neurons: `noise * N` rounded to the nearest integer, halves
    /// away from zero.
    pub fn flips(noise: f64, neurons: usize) -> usize {
        (noise * neurons as f64).round() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    struct Wrapper {
        input: ExperimentInput,
    }

    fn parse(text: &str) -> Result<ExperimentInput, toml::de::Error> {
        toml::from_str::<Wrapper>(text).map(|wrapper| wrapper.input)
    }

    #[test]
    fn parses_both_sources() {
        assert_eq!(
            parse("[input]\nsource = \"pattern\"\nnoise = [0.0, 0.25]").unwrap(),
            ExperimentInput::Pattern {
                noise: vec![0.0, 0.25]
            }
        );
        let random = parse("[input]\nsource = \"random\"").unwrap();
        assert_eq!(random, ExperimentInput::Random {});
        assert_eq!(random.source(), "random");
        assert!(parse("[input]\nsource = \"pattern\"").is_err());
        assert!(parse("[input]\nsource = \"random\"\nnoise = [0.1]").is_err());
        assert!(parse("[input]\nsource = \"file\"").is_err());
    }

    /// `0.1 * 49 = 4.9` rounds to 5, `0.25 * 6 = 1.5` to 2, `0.2 * 49 = 9.8`
    /// to 10.
    #[test]
    fn rounds_the_number_of_flips() {
        assert_eq!(ExperimentInput::flips(0.0, 49), 0);
        assert_eq!(ExperimentInput::flips(0.1, 49), 5);
        assert_eq!(ExperimentInput::flips(0.25, 6), 2);
        assert_eq!(ExperimentInput::flips(0.2, 49), 10);
        assert_eq!(ExperimentInput::flips(1.0, 49), 49);
    }
}
