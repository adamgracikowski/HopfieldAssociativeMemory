//! Configuration file of a `train` or `recall` run.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{DatasetInfo, Input, Learning};
use crate::dynamics::Dynamics;
use crate::paths::{directory_of, resolve};
use crate::patterns::registry::Registry;
use crate::weights::Backend;
use crate::{Error, PatternSet, Result};

/// Settings of a run, read from a TOML file:
///
/// ```toml
/// dataset = "small-7x7"   # name in the dataset registry
/// patterns = 3            # optional: only the first 3 patterns
/// seed = 42               # required by `recall`
/// backend = "auto"        # optional, `auto` by default
///
/// [learning]
/// rule = "hebb"
///
/// [dynamics]              # optional, defaults of `Dynamics`
/// mode = "synchronous"
///
/// [input]                 # required by `recall`
/// source = "pattern"
/// index = 0
/// flips = 5
/// ```
///
/// The registry is `registry` if given, otherwise
/// `data/datasets/datasets.toml` found from the directory of the
/// configuration file upwards. Relative paths (`registry`, the input file)
/// are resolved against the directory of the configuration file when they
/// are used and kept as written in the fields, so the serialized
/// configuration does not depend on the working directory.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Dataset name in the registry.
    pub dataset: String,
    /// Number of leading patterns of the dataset to store; all of them if
    /// not given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patterns: Option<usize>,
    /// Dataset registry file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<PathBuf>,
    /// Seed of the random generator (initial state and update order).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Weight storage backend.
    #[serde(default)]
    pub backend: Backend,
    /// Learning rule.
    pub learning: Learning,
    /// Recall dynamics; its `zero_field` also decides pattern stability
    /// after training.
    #[serde(default)]
    pub dynamics: Dynamics,
    /// Initial state of a recall run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Input>,
    #[serde(skip)]
    path: PathBuf,
}

impl RunConfig {
    /// Reads and parses the configuration file at `path`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parses configuration `text` of the file `path`.
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let mut config: Self = toml::from_str(text).map_err(|source| Error::Config {
            path: path.to_owned(),
            source: Box::new(source),
        })?;
        config.path = path.to_owned();
        Ok(config)
    }

    /// Configuration file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Directory of the configuration file, the base of its relative paths.
    pub fn directory(&self) -> &Path {
        directory_of(&self.path)
    }

    /// Opens the dataset registry: `registry` resolved against
    /// [`directory`](Self::directory) if given, otherwise the one found from
    /// that directory upwards.
    pub fn open_registry(&self) -> Result<Registry> {
        Registry::open(self.registry.as_deref(), self.directory())
    }

    /// Loads the dataset (its first `patterns` patterns, if given) from the
    /// [registry](Self::open_registry).
    pub fn load_dataset(&self) -> Result<(DatasetInfo, PatternSet)> {
        DatasetInfo::load(
            &self.open_registry()?,
            &self.dataset,
            self.patterns,
            self.directory(),
        )
    }

    /// The seed, or an error if the configuration has none.
    pub fn require_seed(&self) -> Result<u64> {
        self.seed.ok_or_else(|| self.missing("`seed`"))
    }

    /// The initial state settings with an input file resolved against
    /// [`directory`](Self::directory), or an error if the configuration has
    /// no `[input]` table.
    pub fn require_input(&self) -> Result<Input> {
        let input = self
            .input
            .clone()
            .ok_or_else(|| self.missing("an `[input]` table"))?;
        Ok(match input {
            Input::File { path, index } => Input::File {
                path: resolve(self.directory(), &path),
                index,
            },
            other => other,
        })
    }

    fn missing(&self, what: &str) -> Error {
        Error::InvalidConfig {
            path: self.path.clone(),
            message: format!("{what} is required for recall"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{Mode, ZeroField};

    const CONFIG: &str = r#"
        dataset = "small-7x7"
        registry = "../datasets/datasets.toml"
        seed = 42
        backend = "dense"

        [learning]
        rule = "hebb"

        [dynamics]
        mode = "synchronous"
        zero_field = "plus-one"

        [input]
        source = "file"
        path = "inputs/a.npy"
    "#;

    fn parse(text: &str) -> Result<RunConfig> {
        RunConfig::parse(text, Path::new("data/experiments/c.toml"))
    }

    #[test]
    fn keeps_paths_as_written_and_resolves_them_on_use() {
        let config = parse(CONFIG).unwrap();
        assert_eq!(config.dataset, "small-7x7");
        assert_eq!(
            config.registry.as_deref(),
            Some(Path::new("../datasets/datasets.toml"))
        );
        assert_eq!(config.directory(), Path::new("data/experiments"));
        assert_eq!(config.require_seed().unwrap(), 42);
        assert_eq!(config.backend, Backend::Dense);
        assert_eq!(config.learning, Learning::Hebb {});
        assert_eq!(config.dynamics.mode, Mode::Synchronous);
        assert_eq!(config.dynamics.zero_field, ZeroField::PlusOne);
        assert_eq!(
            config.require_input().unwrap(),
            Input::File {
                path: PathBuf::from("data/experiments/inputs/a.npy"),
                index: 0
            }
        );
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["registry"], "../datasets/datasets.toml");
        assert_eq!(json["input"]["path"], "inputs/a.npy");
        assert_eq!(config.path(), Path::new("data/experiments/c.toml"));
    }

    #[test]
    fn fills_in_defaults() {
        let config = parse("dataset = \"small-7x7\"\n[learning]\nrule = \"oja\"").unwrap();
        assert_eq!(config.backend, Backend::Auto);
        assert_eq!(config.dynamics, Dynamics::default());
        assert_eq!(
            (config.seed, config.registry.clone(), config.input.clone()),
            (None, None, None)
        );
        let error = config.require_seed().unwrap_err();
        assert!(error.to_string().contains("`seed` is required"), "{error}");
        let error = config.require_input().unwrap_err();
        assert!(error.to_string().contains("`[input]` table"), "{error}");
    }

    #[test]
    fn rejects_invalid_files() {
        for text in [
            "dataset = \"a\"",
            "dataset = \"a\"\nepochs = 3\n[learning]\nrule = \"hebb\"",
            "dataset = \"a\"\n[learning]\nrule = \"perceptron\"",
            "dataset = \"a\"\nseed = -1\n[learning]\nrule = \"hebb\"",
        ] {
            let error = parse(text).unwrap_err();
            assert!(matches!(error, Error::Config { .. }), "{text}: {error}");
        }
    }

    /// The serialized configuration holds every setting, including the
    /// defaults, and reads back to the same values.
    #[test]
    fn round_trips_through_json() {
        let config = parse(CONFIG).unwrap();
        let json = serde_json::to_string(&config).unwrap();
        let mut back: RunConfig = serde_json::from_str(&json).unwrap();
        back.path = config.path.clone();
        assert_eq!(back, config);
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["dynamics"]["order"], "permutation");
        assert_eq!(value["dynamics"]["max_steps"], 1000);
        assert!(value.get("path").is_none());
    }

    #[test]
    fn finds_the_registry_from_the_config_directory() {
        let config = RunConfig::parse(
            "dataset = \"small-7x7\"\n[learning]\nrule = \"hebb\"",
            Path::new("../../data/experiments/00-example/config.toml"),
        )
        .unwrap();
        let registry = config.open_registry().unwrap();
        assert!(registry.names().any(|name| name == "small-7x7"));
    }
}
