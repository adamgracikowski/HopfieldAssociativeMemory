//! Configuration of a search for stable hyperplane sets.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::Hyperplane;
use super::hyperplane::MAX_NEURONS;
use crate::learning::Oja;
use crate::paths::{directory_of, resolve};
use crate::{Error, Result};

/// Settings of a search, read from a TOML file:
///
/// ```toml
/// width = 5            # bitmap of N = width * height neurons, N <= 32
/// height = 5
/// max_weight = 3       # weights of u from 1 to max_weight
/// verify = 5           # largest sets checked for stability
/// samples = 24         # members of the best set written to `sample`
/// basis = "../../datasets/stable-5x5-basis.npy"    # optional outputs
/// sample = "../../datasets/stable-5x5-sample.npy"
///
/// [oja]                # optional, parameters of Oja's rule
/// eta = 0.5
/// ```
///
/// Output files are relative to the directory of the configuration file.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StableConfig {
    /// Bitmap width.
    pub width: usize,
    /// Bitmap height.
    pub height: usize,
    /// Largest weight of the hyperplane vectors `u`.
    pub max_weight: u32,
    /// Number of the largest sets checked for strict stability.
    pub verify: usize,
    /// Number of members of the best set written to `sample`.
    #[serde(default = "default_samples")]
    pub samples: usize,
    /// Oja's rule trained on a basis of every checked set.
    #[serde(default)]
    pub oja: Oja,
    /// `.npy` file for the basis of the best set (the training set of Oja's
    /// rule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis: Option<PathBuf>,
    /// `.npy` file for evenly spread members of the best set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<PathBuf>,
    #[serde(skip)]
    path: PathBuf,
}

/// Largest number of candidate vectors `u` a search builds, all held in
/// memory.
pub const MAX_CANDIDATES: u128 = 10_000_000;

fn default_samples() -> usize {
    24
}

impl StableConfig {
    /// Reads, parses and validates the configuration file at `path`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parses and validates configuration `text` of the file `path`: the
    /// bitmap has 2 to 32 pixels, `max_weight`, `verify` and `samples` are at
    /// least 1, there are at most [`MAX_CANDIDATES`] candidate vectors, and
    /// the two outputs, if given, are different files after resolving them
    /// (`.` components ignored).
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let mut config: Self = toml::from_str(text).map_err(|source| Error::Config {
            path: path.to_owned(),
            source: Box::new(source),
        })?;
        config.path = path.to_owned();
        let neurons = config.neurons();
        if !(2..=MAX_NEURONS).contains(&neurons) {
            return Err(config.invalid(format!(
                "a {} x {} bitmap has {neurons} pixels; the search enumerates 2^N states, so N \
                 must be from 2 to {MAX_NEURONS}",
                config.width, config.height
            )));
        }
        for (name, value) in [
            ("max_weight", config.max_weight as usize),
            ("verify", config.verify),
            ("samples", config.samples),
        ] {
            if value == 0 {
                return Err(config.invalid(format!("`{name}` must be at least 1")));
            }
        }
        let candidates = Hyperplane::candidate_count(neurons, config.max_weight);
        if candidates > MAX_CANDIDATES {
            return Err(config.invalid(format!(
                "`max_weight` = {} gives {candidates} candidates for N = {neurons}; at most {MAX_CANDIDATES} are searched",
                config.max_weight
            )));
        }
        if let (Some(basis), Some(sample)) = (&config.basis, &config.sample)
            && config
                .resolve(basis)
                .components()
                .eq(config.resolve(sample).components())
        {
            return Err(config.invalid("`basis` and `sample` are the same file".to_owned()));
        }
        Ok(config)
    }

    /// Number of neurons `N = width * height`.
    pub fn neurons(&self) -> usize {
        self.width * self.height
    }

    /// Configuration file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Directory of the configuration file, the base of its relative paths.
    pub fn directory(&self) -> &Path {
        directory_of(&self.path)
    }

    /// `file` resolved against the directory of the configuration.
    pub fn resolve(&self, file: &Path) -> PathBuf {
        resolve(self.directory(), file)
    }

    fn invalid(&self, message: String) -> Error {
        Error::InvalidConfig {
            path: self.path.clone(),
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = "width = 4\nheight = 2\nmax_weight = 2\nverify = 3\n\
                          basis = \"basis.npy\"\nsample = \"sample.npy\"\n";

    fn parse(text: &str) -> Result<StableConfig> {
        StableConfig::parse(text, Path::new("../../data/experiments/x/config.toml"))
    }

    #[test]
    fn parses_the_search_and_defaults() {
        let config = parse(CONFIG).unwrap();
        assert_eq!(config.neurons(), 8);
        assert_eq!(config.samples, 24);
        assert_eq!(config.oja, Oja::default());
        assert!(
            config
                .resolve(config.basis.as_ref().unwrap())
                .ends_with("x/basis.npy")
        );
    }

    #[test]
    fn rejects_invalid_settings() {
        let cases = [
            (CONFIG.replace("height = 2", "height = 9"), "from 2 to 32"),
            (
                CONFIG
                    .replace("width = 4", "width = 1\nheight = 1")
                    .replace("height = 2\n", ""),
                "from 2 to 32",
            ),
            (
                CONFIG.replace("max_weight = 2", "max_weight = 0"),
                "`max_weight`",
            ),
            (CONFIG.replace("verify = 3", "verify = 0"), "`verify`"),
            (CONFIG.replace("sample.npy", "basis.npy"), "same file"),
            (
                CONFIG.replace("\"sample.npy\"", "\"./basis.npy\""),
                "same file",
            ),
            (
                CONFIG
                    .replace("height = 2", "height = 4")
                    .replace("max_weight = 2", "max_weight = 20"),
                "candidates",
            ),
            (format!("{CONFIG}samples = 0\n"), "`samples`"),
        ];
        for (text, expected) in cases {
            let message = parse(&text).unwrap_err().to_string();
            assert!(message.contains(expected), "{expected}: {message}");
        }
    }
}
