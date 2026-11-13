//! Configuration file of a benchmark.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::experiment::config::check_axis;
use crate::paths::directory_of;
use crate::patterns::registry::Registry;
use crate::run::Learning;
use crate::weights::Backend;
use crate::{Error, Result};

/// Settings of a benchmark, read from a TOML file:
///
/// ```toml
/// seed = 7
/// datasets = ["cats-300x200-median"]   # optional, registered datasets
/// repeats = 5                         # measurements of every combination
/// backends = ["dense", "factored"]
/// dense_limit = 16384                 # largest N measured with dense weights
/// noise = 0.1                         # flipped fraction of the recall input
///
/// [random]                            # optional, random pattern sets
/// neurons = [1024, 4096, 16384]
/// patterns = 12
///
/// [[learning]]
/// rule = "hebb"
/// [[learning]]
/// rule = "oja"
/// ```
///
/// The registry is `registry` if given (relative to the directory of the
/// configuration file), otherwise `data/datasets/datasets.toml` found from
/// that directory upwards.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkConfig {
    /// Master seed of the random sets, states and update orders.
    pub seed: u64,
    /// Registered datasets.
    #[serde(default)]
    pub datasets: Vec<String>,
    /// Dataset registry file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<PathBuf>,
    /// Random pattern sets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random: Option<RandomSizes>,
    /// Number of measurements of every (set, backend, rule).
    pub repeats: usize,
    /// Weight storage backends, `dense` or `factored`.
    pub backends: Vec<Backend>,
    /// Largest number of neurons measured with dense weights; larger sets
    /// are measured with the other backends only.
    pub dense_limit: usize,
    /// Fraction of the pixels of the stored pattern flipped in the input of
    /// the recall runs.
    pub noise: f64,
    /// Learning rules.
    pub learning: Vec<Learning>,
    #[serde(skip)]
    path: PathBuf,
}

/// Random sets of `patterns` patterns for every number of neurons in
/// `neurons`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RandomSizes {
    /// Numbers of neurons `N`.
    pub neurons: Vec<usize>,
    /// Number of patterns `p`.
    pub patterns: usize,
}

impl BenchmarkConfig {
    /// Reads, parses and validates the configuration file at `path`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parses and [validates](Self::validate) configuration `text` of the
    /// file `path`.
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let mut config: Self = toml::from_str(text).map_err(|source| Error::Config {
            path: path.to_owned(),
            source: Box::new(source),
        })?;
        config.path = path.to_owned();
        config.validate()?;
        Ok(config)
    }

    /// Checks that there is at least one dataset or random size, no repeated
    /// dataset, number of neurons, backend or learning rule, at least one
    /// repeat, only the `dense` and `factored` backends, a noise level in
    /// `[0, 1]`, and random sets of at least one and at most `N` patterns.
    pub fn validate(&self) -> Result<()> {
        let neurons = self
            .random
            .as_ref()
            .map_or(&[][..], |random| &random.neurons[..]);
        if self.datasets.is_empty() && neurons.is_empty() {
            return Err(
                self.invalid("`datasets` or `random.neurons` needs at least one value".to_owned())
            );
        }
        if !self.datasets.is_empty() {
            check_axis("datasets", &self.datasets).map_err(|m| self.invalid(m))?;
        }
        if !neurons.is_empty() {
            check_axis("random.neurons", neurons).map_err(|m| self.invalid(m))?;
        }
        check_axis("backends", &self.backends).map_err(|m| self.invalid(m))?;
        if self.backends.contains(&Backend::Auto) {
            return Err(self.invalid(
                "`backends` takes `dense` and `factored`; `auto` would measure one of them twice"
                    .to_owned(),
            ));
        }
        check_axis("learning", &self.learning_labels()).map_err(|m| self.invalid(m))?;
        if self.repeats == 0 {
            return Err(self.invalid("`repeats` must be at least 1".to_owned()));
        }
        if !(0.0..=1.0).contains(&self.noise) {
            return Err(self.invalid(format!("`noise` {} is not in [0, 1]", self.noise)));
        }
        if let Some(random) = &self.random {
            if random.patterns == 0 {
                return Err(self.invalid("`random.patterns` must be at least 1".to_owned()));
            }
            if let Some(&n) = neurons.iter().find(|&&n| n < random.patterns) {
                return Err(self.invalid(format!(
                    "random sets of N = {n} neurons cannot have {} patterns (at most N)",
                    random.patterns
                )));
            }
        }
        Ok(())
    }

    /// Whether a set of `neurons` neurons is measured with `backend`: every
    /// backend except dense, which only up to [`dense_limit`](Self::dense_limit).
    pub fn measures(&self, backend: Backend, neurons: usize) -> bool {
        backend != Backend::Dense || neurons <= self.dense_limit
    }

    /// Checks that at least one set of `neurons` (the numbers of neurons of
    /// all sets, datasets included) is [measured](Self::measures) with at
    /// least one backend, e.g. not only dense weights for sets larger than
    /// `dense_limit`.
    pub fn check_measured(&self, neurons: &[usize]) -> Result<()> {
        let any = neurons.iter().any(|&n| {
            self.backends
                .iter()
                .any(|&backend| self.measures(backend, n))
        });
        if any {
            Ok(())
        } else {
            Err(self.invalid(format!(
                "no set is measured: the sets have {neurons:?} neurons, above `dense_limit` = {} \
                 of the only backend `dense`",
                self.dense_limit
            )))
        }
    }

    /// Configuration file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Directory of the configuration file, the base of its relative paths.
    pub fn directory(&self) -> &Path {
        directory_of(&self.path)
    }

    /// Opens the dataset registry (see [`Registry::open`]).
    pub fn open_registry(&self) -> Result<Registry> {
        Registry::open(self.registry.as_deref(), self.directory())
    }

    /// Names of the learning rules, e.g. `hebb` or `oja (subspace)`.
    pub fn learning_labels(&self) -> Vec<String> {
        self.learning.iter().map(ToString::to_string).collect()
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

    const CONFIG: &str = r#"
        seed = 7
        datasets = ["small-7x7"]
        repeats = 2
        backends = ["dense", "factored"]
        dense_limit = 64
        noise = 0.1

        [random]
        neurons = [64, 128]
        patterns = 3

        [[learning]]
        rule = "hebb"
        [[learning]]
        rule = "oja"
    "#;

    fn parse(text: &str) -> Result<BenchmarkConfig> {
        BenchmarkConfig::parse(text, Path::new("../../data/experiments/x/config.toml"))
    }

    #[test]
    fn parses_sets_backends_and_rules() {
        let config = parse(CONFIG).unwrap();
        assert_eq!(config.datasets, ["small-7x7"]);
        assert_eq!(config.random.as_ref().unwrap().neurons, [64, 128]);
        assert_eq!(config.backends, [Backend::Dense, Backend::Factored]);
        assert_eq!(config.learning_labels(), ["hebb", "oja (subspace)"]);
        assert!(config.measures(Backend::Dense, 64));
        assert!(!config.measures(Backend::Dense, 65));
        assert!(config.measures(Backend::Factored, 1_000_000));
    }

    /// With dense weights only and `dense_limit = 64`, sets of 49 and 128
    /// neurons give a measurement, sets of 128 neurons alone none.
    #[test]
    fn rejects_a_benchmark_that_measures_nothing() {
        let config = parse(&CONFIG.replace("[\"dense\", \"factored\"]", "[\"dense\"]")).unwrap();
        assert!(config.check_measured(&[49, 128]).is_ok());
        let message = config.check_measured(&[128]).unwrap_err().to_string();
        assert!(message.contains("no set is measured"), "{message}");
    }

    #[test]
    fn rejects_invalid_settings() {
        let cases = [
            (
                CONFIG
                    .replace("datasets = [\"small-7x7\"]", "")
                    .replace("neurons = [64, 128]", "neurons = []"),
                "at least one value",
            ),
            (CONFIG.replace("[64, 128]", "[64, 64]"), "twice"),
            (
                CONFIG.replace("[\"dense\", \"factored\"]", "[\"dense\", \"dense\"]"),
                "twice",
            ),
            (
                CONFIG.replace("[\"dense\", \"factored\"]", "[\"auto\"]"),
                "`auto`",
            ),
            (CONFIG.replace("repeats = 2", "repeats = 0"), "`repeats`"),
            (CONFIG.replace("noise = 0.1", "noise = 1.5"), "`noise`"),
            (CONFIG.replace("patterns = 3", "patterns = 0"), "at least 1"),
            (
                CONFIG.replace("patterns = 3", "patterns = 100"),
                "at most N",
            ),
            (CONFIG.replace("rule = \"oja\"", "rule = \"hebb\""), "twice"),
        ];
        for (text, expected) in cases {
            let message = parse(&text).unwrap_err().to_string();
            assert!(message.contains(expected), "{expected}: {message}");
        }
    }
}
