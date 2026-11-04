//! Initial state of a recall run.

use std::path::PathBuf;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::patterns::Pattern;
use crate::patterns::npy::read_npy;
use crate::{Error, PatternSet, Result};

/// Where the initial state of a recall run comes from, written as an
/// `[input]` table whose `source` key names the kind:
///
/// ```toml
/// [input]
/// source = "pattern" # stored pattern `index` with `flips` random pixels flipped
/// index = 2
/// flips = 10
/// ```
///
/// `source = "random"` takes no parameters; `source = "file"` takes `path`
/// (an `.npy` pattern set of the same bitmap size, relative to the
/// configuration file) and `index` (0 by default).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Input {
    /// Stored pattern `index` with exactly `flips` pixels flipped, chosen at
    /// random without repetition.
    Pattern {
        /// Index of the stored pattern, from 0.
        index: usize,
        /// Number of flipped pixels.
        #[serde(default)]
        flips: usize,
    },
    /// A random state, every pixel `1` or `-1` with probability 1/2.
    Random {},
    /// Pattern `index` of an `.npy` pattern set.
    File {
        /// The `.npy` file.
        path: PathBuf,
        /// Index of the pattern in the file, from 0.
        #[serde(default)]
        index: usize,
    },
}

/// Initial state of a run and the stored pattern it was derived from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Initial {
    /// The initial state.
    pub state: Pattern,
    /// Index of the stored pattern the state was made from, if any.
    pub target: Option<usize>,
}

impl Input {
    /// Builds the initial state for a network that stores `set`. Random
    /// choices come from `rng`.
    pub fn initial<R: Rng + ?Sized>(&self, set: &PatternSet, rng: &mut R) -> Result<Initial> {
        match self {
            Self::Pattern { index, flips } => {
                check_index(*index, set.len(), "the dataset")?;
                Ok(Initial {
                    state: set.pattern(*index).flipped(*flips, rng)?,
                    target: Some(*index),
                })
            }
            Self::Random {} => Ok(Initial {
                state: Pattern::random(set.neurons(), rng)?,
                target: None,
            }),
            Self::File { path, index } => {
                let file = read_npy(path)?;
                if (file.width(), file.height()) != (set.width(), set.height()) {
                    return Err(Error::InvalidArgument {
                        message: format!(
                            "input `{}` has {}x{} bitmaps, the dataset {}x{}",
                            path.display(),
                            file.width(),
                            file.height(),
                            set.width(),
                            set.height()
                        ),
                    });
                }
                check_index(*index, file.len(), &format!("`{}`", path.display()))?;
                Ok(Initial {
                    state: file.pattern(*index).to_pattern(),
                    target: None,
                })
            }
        }
    }
}

fn check_index(index: usize, len: usize, owner: &str) -> Result<()> {
    if index >= len {
        return Err(Error::InvalidArgument {
            message: format!("pattern {index} does not exist, {owner} has {len} patterns"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::npy::write_npy;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn set() -> PatternSet {
        PatternSet::new(3, 2, array![[1, 1, 1, -1, -1, -1], [1, -1, 1, -1, 1, -1]]).unwrap()
    }

    fn initial(input: &Input, seed: u64) -> Result<Initial> {
        input.initial(&set(), &mut ChaCha8Rng::seed_from_u64(seed))
    }

    #[test]
    fn pattern_with_flipped_pixels() {
        let input = Input::Pattern { index: 1, flips: 2 };
        let first = initial(&input, 3).unwrap();
        assert_eq!(first.target, Some(1));
        assert_eq!(first.state.hamming(&set().pattern(1)), 2);
        assert_eq!(first, initial(&input, 3).unwrap());

        let exact = initial(&Input::Pattern { index: 0, flips: 0 }, 0).unwrap();
        assert_eq!(exact.state, set().pattern(0));
        let error = initial(&Input::Pattern { index: 2, flips: 0 }, 0).unwrap_err();
        assert!(error.to_string().contains("has 2 patterns"), "{error}");
        assert!(initial(&Input::Pattern { index: 0, flips: 7 }, 0).is_err());
    }

    #[test]
    fn random_state_has_n_pixels() {
        let random = initial(&Input::Random {}, 1).unwrap();
        assert_eq!(random.state.len(), 6);
        assert_eq!(random.target, None);
    }

    #[test]
    fn pattern_from_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("inputs.npy");
        let inputs =
            PatternSet::new(3, 2, array![[1, 1, 1, 1, 1, 1], [-1, 1, -1, 1, -1, 1]]).unwrap();
        write_npy(&inputs, &path).unwrap();
        let input = Input::File {
            path: path.clone(),
            index: 1,
        };
        let state = initial(&input, 0).unwrap();
        assert_eq!(state.state, inputs.pattern(1));
        assert_eq!(state.target, None);
        let error = initial(
            &Input::File {
                path: path.clone(),
                index: 2,
            },
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("has 2 patterns"), "{error}");

        let other = PatternSet::new(2, 3, inputs.patterns().to_owned()).unwrap();
        write_npy(&other, &path).unwrap();
        let error = initial(&input, 0).unwrap_err();
        assert!(error.to_string().contains("2x3 bitmaps"), "{error}");
    }

    #[test]
    fn parses_every_source() {
        #[derive(Deserialize)]
        struct Wrapper {
            input: Input,
        }
        let parse = |text: &str| toml::from_str::<Wrapper>(text).map(|w| w.input);
        assert_eq!(
            parse("[input]\nsource = \"pattern\"\nindex = 3").unwrap(),
            Input::Pattern { index: 3, flips: 0 }
        );
        assert_eq!(
            parse("[input]\nsource = \"random\"").unwrap(),
            Input::Random {}
        );
        assert_eq!(
            parse("[input]\nsource = \"file\"\npath = \"x.npy\"").unwrap(),
            Input::File {
                path: PathBuf::from("x.npy"),
                index: 0
            }
        );
        assert!(parse("[input]\nsource = \"random\"\nflips = 2").is_err());
        assert!(parse("[input]\nsource = \"pattern\"").is_err());
    }
}
