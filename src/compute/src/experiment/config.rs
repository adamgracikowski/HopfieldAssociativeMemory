//! Configuration file of an experiment.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{Cell, ExperimentDynamics, ExperimentInput};
use crate::paths::directory_of;
use crate::patterns::registry::Registry;
use crate::run::Learning;
use crate::weights::Backend;
use crate::{Error, Result};

/// Settings of an experiment, read from a TOML file. The grid is the product
/// of the axes `datasets`, `learning`, `dynamics.modes` and `input.noise`;
/// each of its cells runs `trials` trials:
///
/// ```toml
/// seed = 7
/// datasets = ["small-7x7", "animals-14x9"]
/// backend = "auto"         # optional, `auto` by default
/// trials = 100
///
/// [[learning]]
/// rule = "hebb"
/// [[learning]]
/// rule = "oja"
///
/// [dynamics]               # optional, see `ExperimentDynamics`
/// modes = ["synchronous", "asynchronous"]
///
/// [input]
/// source = "pattern"
/// noise = [0.0, 0.1, 0.2]
/// ```
///
/// The registry is `registry` if given (relative to the directory of the
/// configuration file), otherwise `data/datasets/datasets.toml` found from
/// that directory upwards.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    /// Master seed; every trial derives its own seed from it.
    pub seed: u64,
    /// Dataset names in the registry, one grid axis.
    pub datasets: Vec<String>,
    /// Dataset registry file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<PathBuf>,
    /// Weight storage backend of every network.
    #[serde(default)]
    pub backend: Backend,
    /// Number of trials per cell.
    pub trials: usize,
    /// Learning rules, one grid axis.
    pub learning: Vec<Learning>,
    /// Update modes and shared dynamics settings.
    #[serde(default)]
    pub dynamics: ExperimentDynamics,
    /// Initial states and noise levels.
    pub input: ExperimentInput,
    #[serde(skip)]
    path: PathBuf,
}

impl ExperimentConfig {
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

    /// Checks that every axis is non-empty and has no repeated values, that
    /// noise levels lie in `[0, 1]`, that there is at least one trial and
    /// one step, and that the learning rules have different names (their
    /// [`Display`](std::fmt::Display) identifies them in the results).
    pub fn validate(&self) -> Result<()> {
        let labels = self.learning_labels();
        check_axis("datasets", &self.datasets).map_err(|m| self.invalid(m))?;
        check_axis("learning", &labels).map_err(|m| self.invalid(m))?;
        check_axis("dynamics.modes", &self.dynamics.modes).map_err(|m| self.invalid(m))?;
        if let ExperimentInput::Pattern { noise } = &self.input {
            check_axis("input.noise", noise).map_err(|m| self.invalid(m))?;
            if let Some(level) = noise.iter().find(|level| !(0.0..=1.0).contains(*level)) {
                return Err(self.invalid(format!(
                    "noise level {level} is not a fraction between 0 and 1"
                )));
            }
        }
        if self.trials == 0 {
            return Err(self.invalid("`trials` must be at least 1".to_owned()));
        }
        if self.dynamics.max_steps == 0 {
            return Err(self.invalid("`dynamics.max_steps` must be at least 1".to_owned()));
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

    /// Noise levels of the grid: the configured ones, or a single `None`
    /// for random initial states.
    pub fn noise_levels(&self) -> Vec<Option<f64>> {
        match &self.input {
            ExperimentInput::Pattern { noise } => noise.iter().copied().map(Some).collect(),
            ExperimentInput::Random {} => vec![None],
        }
    }

    /// Every cell of the grid, ordered by dataset, learning rule, mode and
    /// noise level (the last axis changes fastest).
    pub fn cells(&self) -> Vec<Cell> {
        let noise_levels = self.noise_levels();
        let mut cells = Vec::new();
        for dataset in 0..self.datasets.len() {
            for learning in 0..self.learning.len() {
                for &mode in &self.dynamics.modes {
                    for &noise in &noise_levels {
                        cells.push(Cell {
                            dataset,
                            learning,
                            mode,
                            noise,
                        });
                    }
                }
            }
        }
        cells
    }

    fn invalid(&self, message: String) -> Error {
        Error::InvalidConfig {
            path: self.path.clone(),
            message,
        }
    }
}

/// Checks that the axis `name` has at least one value and no value twice.
fn check_axis<T: PartialEq + std::fmt::Debug>(
    name: &str,
    values: &[T],
) -> std::result::Result<(), String> {
    if values.is_empty() {
        return Err(format!("`{name}` needs at least one value"));
    }
    for (i, value) in values.iter().enumerate() {
        if values[..i].contains(value) {
            return Err(format!("`{name}` lists {value:?} twice"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{Mode, ZeroField};
    use crate::learning::Oja;

    const CONFIG: &str = r#"
        seed = 7
        datasets = ["small-7x7", "animals-14x9"]
        trials = 20

        [[learning]]
        rule = "hebb"
        [[learning]]
        rule = "oja"

        [dynamics]
        modes = ["synchronous", "asynchronous"]

        [input]
        source = "pattern"
        noise = [0.0, 0.1, 0.2]
    "#;

    fn parse(text: &str) -> Result<ExperimentConfig> {
        ExperimentConfig::parse(text, Path::new("data/experiments/00-example/config.toml"))
    }

    fn invalid_message(text: &str) -> String {
        match parse(text) {
            Err(Error::InvalidConfig { message, .. }) => message,
            other => panic!("expected an invalid configuration, got {other:?}"),
        }
    }

    #[test]
    fn parses_an_experiment_with_defaults() {
        let config = parse(CONFIG).unwrap();
        assert_eq!(config.seed, 7);
        assert_eq!(config.datasets, ["small-7x7", "animals-14x9"]);
        assert_eq!(config.backend, Backend::Auto);
        assert_eq!(config.trials, 20);
        assert_eq!(
            config.learning,
            [Learning::Hebb {}, Learning::Oja(Oja::default())]
        );
        assert_eq!(config.learning_labels(), ["hebb", "oja (subspace)"]);
        assert_eq!(
            config.dynamics.modes,
            [Mode::Synchronous, Mode::Asynchronous]
        );
        assert_eq!(config.dynamics.zero_field, ZeroField::Keep);
        assert_eq!(config.dynamics.max_steps, 1000);
        assert_eq!(config.noise_levels(), [Some(0.0), Some(0.1), Some(0.2)]);
        assert_eq!(config.directory(), Path::new("data/experiments/00-example"));
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["dynamics"]["order"], "permutation");
        assert_eq!(json["input"]["source"], "pattern");
        assert!(json.get("registry").is_none());
    }

    /// 2 datasets x 2 rules x 2 modes x 3 noise levels: 24 cells, the noise
    /// level changing fastest.
    #[test]
    fn the_grid_is_the_product_of_the_axes() {
        let config = parse(CONFIG).unwrap();
        let cells = config.cells();
        assert_eq!(cells.len(), 24);
        assert_eq!(
            cells[0],
            Cell {
                dataset: 0,
                learning: 0,
                mode: Mode::Synchronous,
                noise: Some(0.0)
            }
        );
        assert_eq!(cells[1].noise, Some(0.1));
        assert_eq!(cells[3].mode, Mode::Asynchronous);
        assert_eq!(cells[6].learning, 1);
        assert_eq!(cells[12].dataset, 1);
        let random = parse(&CONFIG.replace(
            "source = \"pattern\"\n        noise = [0.0, 0.1, 0.2]",
            "source = \"random\"",
        ))
        .unwrap();
        assert_eq!(random.noise_levels(), [None]);
        assert_eq!(random.cells().len(), 8);
    }

    #[test]
    fn rejects_invalid_values() {
        let cases = [
            (
                "datasets = [\"small-7x7\", \"animals-14x9\"]",
                "datasets = []",
                "`datasets` needs",
            ),
            (
                "datasets = [\"small-7x7\", \"animals-14x9\"]",
                "datasets = [\"a\", \"a\"]",
                "`datasets` lists \"a\" twice",
            ),
            (
                "rule = \"oja\"",
                "rule = \"hebb\"",
                "`learning` lists \"hebb\" twice",
            ),
            (
                "modes = [\"synchronous\", \"asynchronous\"]",
                "modes = []",
                "`dynamics.modes` needs",
            ),
            (
                "noise = [0.0, 0.1, 0.2]",
                "noise = [0.1, 0.1]",
                "`input.noise` lists 0.1",
            ),
            (
                "noise = [0.0, 0.1, 0.2]",
                "noise = [0.0, 1.5]",
                "noise level 1.5",
            ),
            (
                "noise = [0.0, 0.1, 0.2]",
                "noise = [-0.1]",
                "noise level -0.1",
            ),
            ("trials = 20", "trials = 0", "`trials` must be"),
            (
                "modes = [\"synchronous\", \"asynchronous\"]",
                "modes = [\"synchronous\"]\nmax_steps = 0",
                "`dynamics.max_steps`",
            ),
        ];
        for (from, to, expected) in cases {
            let message = invalid_message(&CONFIG.replace(from, to));
            assert!(message.contains(expected), "{to}: {message}");
        }
    }

    #[test]
    fn rejects_unknown_keys_and_missing_values() {
        for (from, to) in [
            ("trials = 20", "trials = 20\nrepeats = 3"),
            ("seed = 7", ""),
            ("trials = 20", ""),
            ("modes = [", "mode = \"synchronous\"\nmodes = ["),
        ] {
            let error = parse(&CONFIG.replace(from, to)).unwrap_err();
            assert!(matches!(error, Error::Config { .. }), "{to}: {error}");
        }
    }

    #[test]
    fn finds_the_registry_from_the_config_directory() {
        let config = ExperimentConfig::parse(
            CONFIG,
            Path::new("../../data/experiments/00-example/config.toml"),
        )
        .unwrap();
        let registry = config.open_registry().unwrap();
        assert!(registry.names().any(|name| name == "animals-14x9"));
    }
}
