//! Configuration file of a capacity run.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::dynamics::ZeroField;
use crate::experiment::config::check_axis;
use crate::paths::directory_of;
use crate::patterns::registry::Registry;
use crate::run::Learning;
use crate::weights::Backend;
use crate::{Error, Result};

/// Settings of a capacity run, read from a TOML file:
///
/// ```toml
/// seed = 7
/// datasets = ["letters-14x20"]   # prefixes p = 1, 2, ..., len
/// trials = 50                    # random sets per (N, p)
/// zero_field = "keep"            # optional, `keep` by default
/// backend = "auto"               # optional, `auto` by default
///
/// [[random]]
/// neurons = 280
/// patterns = { start = 2, end = 80, step = 2 }
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
pub struct CapacityConfig {
    /// Master seed; every random set derives its own seed from it.
    pub seed: u64,
    /// Registered datasets whose prefixes are checked.
    #[serde(default)]
    pub datasets: Vec<String>,
    /// Dataset registry file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<PathBuf>,
    /// Weight storage backend of every network.
    #[serde(default)]
    pub backend: Backend,
    /// Convention for `sgn(0)` in the stability check.
    #[serde(default)]
    pub zero_field: ZeroField,
    /// Number of random sets for every number of neurons and patterns.
    pub trials: usize,
    /// Random pattern sets.
    #[serde(default)]
    pub random: Vec<RandomSets>,
    /// Learning rules.
    pub learning: Vec<Learning>,
    #[serde(skip)]
    path: PathBuf,
}

/// Random sets of `neurons` neurons for every number of patterns in
/// `patterns`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RandomSets {
    /// Number of neurons `N`.
    pub neurons: usize,
    /// Numbers of patterns `p`.
    pub patterns: PatternRange,
}

/// The numbers `start, start + step, ...` up to and including `end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PatternRange {
    /// First value.
    pub start: usize,
    /// Last value, reached only if `end - start` is a multiple of `step`.
    pub end: usize,
    /// Distance between values, 1 by default.
    #[serde(default = "one")]
    pub step: usize,
}

fn one() -> usize {
    1
}

impl PatternRange {
    /// The values of the range in increasing order.
    pub fn values(&self) -> Vec<usize> {
        (self.start..=self.end).step_by(self.step.max(1)).collect()
    }
}

impl CapacityConfig {
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

    /// Checks that there is at least one dataset or random set, no repeated
    /// dataset, number of neurons or learning rule name, at least one
    /// trial, and that every pattern range is non-empty, starts at 1 or
    /// more, has a positive step and does not exceed its number of neurons.
    pub fn validate(&self) -> Result<()> {
        if self.datasets.is_empty() && self.random.is_empty() {
            return Err(self.invalid("`datasets` or `random` needs at least one value".to_owned()));
        }
        if !self.datasets.is_empty() {
            check_axis("datasets", &self.datasets).map_err(|m| self.invalid(m))?;
        }
        let neurons: Vec<usize> = self.random.iter().map(|sets| sets.neurons).collect();
        if !neurons.is_empty() {
            check_axis("random.neurons", &neurons).map_err(|m| self.invalid(m))?;
        }
        check_axis("learning", &self.learning_labels()).map_err(|m| self.invalid(m))?;
        if self.trials == 0 {
            return Err(self.invalid("`trials` must be at least 1".to_owned()));
        }
        for sets in &self.random {
            let PatternRange { start, end, step } = sets.patterns;
            if start == 0 || step == 0 || end < start {
                return Err(self.invalid(format!(
                    "random patterns {start} to {end} by {step} for N = {} is not a range of \
                     positive numbers",
                    sets.neurons
                )));
            }
            if end > sets.neurons {
                return Err(self.invalid(format!(
                    "random sets of N = {} neurons cannot have {end} patterns (at most N)",
                    sets.neurons
                )));
            }
        }
        Ok(())
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
        trials = 3

        [[random]]
        neurons = 49
        patterns = { start = 2, end = 10, step = 4 }

        [[learning]]
        rule = "hebb"
        [[learning]]
        rule = "oja"
    "#;

    fn parse(text: &str) -> Result<CapacityConfig> {
        CapacityConfig::parse(text, Path::new("../../data/experiments/x/config.toml"))
    }

    fn message(text: &str) -> String {
        parse(text).unwrap_err().to_string()
    }

    #[test]
    fn parses_sources_and_defaults() {
        let config = parse(CONFIG).unwrap();
        assert_eq!(config.datasets, ["small-7x7"]);
        assert_eq!(config.random[0].patterns.values(), [2, 6, 10]);
        assert_eq!(config.zero_field, ZeroField::Keep);
        assert_eq!(config.backend, Backend::Auto);
        assert_eq!(config.learning_labels(), ["hebb", "oja (subspace)"]);
    }

    #[test]
    fn range_step_defaults_to_one_and_may_miss_the_end() {
        let range: PatternRange = toml::from_str("start = 3\nend = 5").unwrap();
        assert_eq!(range.values(), [3, 4, 5]);
        let range = PatternRange {
            start: 1,
            end: 6,
            step: 2,
        };
        assert_eq!(range.values(), [1, 3, 5]);
    }

    #[test]
    fn rejects_invalid_settings() {
        let cases = [
            (
                CONFIG.replace("datasets = [\"small-7x7\"]", "").replace(
                    "[[random]]\n        neurons = 49\n        patterns = { start = 2, end = 10, step = 4 }",
                    "",
                ),
                "at least one value",
            ),
            (
                CONFIG.replace("[\"small-7x7\"]", "[\"small-7x7\", \"small-7x7\"]"),
                "twice",
            ),
            (CONFIG.replace("trials = 3", "trials = 0"), "`trials`"),
            (CONFIG.replace("end = 10", "end = 50"), "at most N"),
            (CONFIG.replace("start = 2", "start = 0"), "positive numbers"),
            (CONFIG.replace("step = 4", "step = 0"), "positive numbers"),
            (
                CONFIG.replace("rule = \"oja\"", "rule = \"hebb\""),
                "twice",
            ),
        ];
        for (text, expected) in cases {
            let message = message(&text);
            assert!(message.contains(expected), "{expected}: {message}");
        }
    }
}
