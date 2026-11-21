//! Initial states of the trials of an experiment.

use serde::{Deserialize, Serialize};

use super::Damage;

/// The `[input]` table of an experiment: the kind of initial state (`source`)
/// and its damage levels, one grid axis.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ExperimentInput {
    /// In every trial each stored pattern in turn, with
    /// [`flips`](Self::flips) random pixels flipped.
    Pattern {
        /// Fractions of flipped pixels, from 0 to 1.
        noise: Vec<f64>,
    },
    /// In every trial each stored pattern in turn, with a band of
    /// [`rows`](Self::rows) consecutive full-width rows at a random position
    /// set to `-1`.
    Occlusion {
        /// Fractions of covered rows, from 0 to 1.
        occlusion: Vec<f64>,
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
            Self::Occlusion { .. } => "occlusion",
            Self::Random {} => "random",
        }
    }

    /// Name of the damage levels as written in the configuration (`noise`
    /// or `occlusion`) and the levels, or `None` for random initial states.
    pub fn levels(&self) -> Option<(&'static str, &[f64])> {
        match self {
            Self::Pattern { noise } => Some(("noise", noise)),
            Self::Occlusion { occlusion } => Some(("occlusion", occlusion)),
            Self::Random {} => None,
        }
    }

    /// The damage at `level` of a `width x height` bitmap, or `None` for
    /// random initial states.
    pub fn damage(&self, level: f64, width: usize, height: usize) -> Option<Damage> {
        match self {
            Self::Pattern { .. } => Some(Damage::Flip {
                pixels: Self::flips(level, width * height),
            }),
            Self::Occlusion { .. } => {
                let rows = Self::rows(level, height);
                Some(Damage::Occlusion {
                    rows,
                    pixels: rows * width,
                })
            }
            Self::Random {} => None,
        }
    }

    /// Number of pixels flipped at noise level `noise`: `noise * N` rounded.
    pub fn flips(noise: f64, neurons: usize) -> usize {
        (noise * neurons as f64).round() as usize
    }

    /// Number of rows covered at occlusion level `occlusion`:
    /// `occlusion * height` rounded.
    pub fn rows(occlusion: f64, height: usize) -> usize {
        (occlusion * height as f64).round() as usize
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
        let occlusion = parse("[input]\nsource = \"occlusion\"\nocclusion = [0.5]").unwrap();
        assert_eq!(
            occlusion,
            ExperimentInput::Occlusion {
                occlusion: vec![0.5]
            }
        );
        assert_eq!(occlusion.source(), "occlusion");
        assert!(parse("[input]\nsource = \"occlusion\"\nnoise = [0.5]").is_err());
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

    /// Levels `0.15, 0.3, 0.45, 0.6, 0.75` of 7 rows are
    /// `1.05, 2.1, 3.15, 4.2, 5.25`, rounded to 1-5 rows; of a 25 x 25
    /// bitmap `0.3 * 25 = 7.5` rounds to 8 rows, 200 pixels.
    #[test]
    fn rounds_the_number_of_rows() {
        let rows: Vec<usize> = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 1.0]
            .iter()
            .map(|&level| ExperimentInput::rows(level, 7))
            .collect();
        assert_eq!(rows, [0, 1, 2, 3, 4, 5, 7]);
        let occlusion = ExperimentInput::Occlusion {
            occlusion: vec![0.3],
        };
        assert_eq!(
            occlusion.damage(0.3, 25, 25),
            Some(Damage::Occlusion {
                rows: 8,
                pixels: 200
            })
        );
        let noise = ExperimentInput::Pattern { noise: vec![0.1] };
        assert_eq!(noise.damage(0.1, 7, 7), Some(Damage::Flip { pixels: 5 }));
        assert_eq!(ExperimentInput::Random {}.damage(0.1, 7, 7), None);
        assert_eq!(noise.levels(), Some(("noise", &[0.1][..])));
        assert_eq!(ExperimentInput::Random {}.levels(), None);
    }
}
