//! Configuration of a generation of grouped pattern sets.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{GroupSet, GroupsOutput};
use crate::experiment::config::check_axis;
use crate::paths::{directory_of, resolve};
use crate::patterns::registry::Registry;
use crate::seed::Seed;
use crate::{Error, PatternSet, Result};

/// Settings of a generation, read from a TOML file:
///
/// ```toml
/// seed = 20261110
/// dataset = "large-25x25"   # prototypes come from this registered dataset
/// prototypes = [0, 2, 4]    # indices of the prototypes, one group each
/// variants = 5              # patterns per group
///
/// [[outputs]]
/// flip = 0.05               # fraction of flipped pixels of every variant
/// file = "../../datasets/groups-25x25-d05.npy"
/// ```
///
/// Every output holds `prototypes.len() * variants` patterns, group by
/// group. Variant `v` of prototype `mu` in output `<stem>.npy` flips
/// `round(flip * N)` pixels chosen with the input stream of the seed derived
/// from `seed` and the text `<stem>/<mu>/<v>`. The prototypes themselves are
/// not stored. The registry is `registry` if given (relative to the
/// directory of the configuration file), otherwise
/// `data/datasets/datasets.toml` found from that directory upwards; output
/// files are relative to the directory of the configuration file.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GroupsConfig {
    /// Master seed of the variants.
    pub seed: u64,
    /// Registered dataset with the prototypes.
    pub dataset: String,
    /// Dataset registry file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<PathBuf>,
    /// Indices of the prototypes in the dataset, one group each.
    pub prototypes: Vec<usize>,
    /// Number of variants of every prototype.
    pub variants: usize,
    /// Generated datasets, as written in the file.
    pub outputs: Vec<GroupsOutput>,
    #[serde(skip)]
    path: PathBuf,
}

impl GroupsConfig {
    /// Reads, parses and validates the configuration file at `path`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parses and validates configuration `text` of the file `path`: there
    /// must be at least one prototype, none twice, at least one variant and
    /// one output, every `flip` must lie in `[0, 1]` and no two outputs may
    /// share a file name, which names their seeds.
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let mut config: Self = toml::from_str(text).map_err(|source| Error::Config {
            path: path.to_owned(),
            source: Box::new(source),
        })?;
        config.path = path.to_owned();
        let invalid = |message: String| Error::InvalidConfig {
            path: path.to_owned(),
            message,
        };
        check_axis("prototypes", &config.prototypes).map_err(invalid)?;
        if config.variants == 0 {
            return Err(invalid("`variants` must be at least 1".to_owned()));
        }
        if config.outputs.is_empty() {
            return Err(invalid("no outputs".to_owned()));
        }
        if let Some(output) = config
            .outputs
            .iter()
            .find(|output| !(0.0..=1.0).contains(&output.flip))
        {
            return Err(invalid(format!(
                "flip {} is not a fraction between 0 and 1",
                output.flip
            )));
        }
        let mut stems = HashSet::new();
        if let Some(output) = config
            .outputs
            .iter()
            .find(|output| !stems.insert(output.stem()))
        {
            return Err(invalid(format!(
                "two outputs share the file name of `{}`",
                output.file.display()
            )));
        }
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

    /// Output file of `output`, resolved against
    /// [`directory`](Self::directory).
    pub fn output_file(&self, output: &GroupsOutput) -> PathBuf {
        resolve(self.directory(), &output.file)
    }

    /// Loads the prototypes from the registry and generates every output.
    /// Nothing is written.
    pub fn generate(&self) -> Result<Vec<GroupSet>> {
        let registry = Registry::open(self.registry.as_deref(), self.directory())?;
        let source = registry.load(&self.dataset)?;
        if let Some(&mu) = self.prototypes.iter().find(|&&mu| mu >= source.len()) {
            return Err(Error::InvalidConfig {
                path: self.path.clone(),
                message: format!(
                    "prototype {mu} does not exist, `{}` has {} patterns",
                    self.dataset,
                    source.len()
                ),
            });
        }
        self.outputs
            .iter()
            .map(|output| self.generate_output(&source, output))
            .collect()
    }

    fn generate_output(&self, source: &PatternSet, output: &GroupsOutput) -> Result<GroupSet> {
        let flips = output.flips(source.neurons());
        let stem = output.stem();
        let mut patterns = Vec::new();
        for &mu in &self.prototypes {
            for v in 0..self.variants {
                let seed = Seed::derive(self.seed, &[&stem, &mu, &v]);
                patterns.push(source.pattern(mu).flipped(flips, &mut seed.input_rng())?);
            }
        }
        let set = PatternSet::from_patterns(source.width(), source.height(), &patterns)?;
        Ok(GroupSet {
            output: GroupsOutput {
                file: self.output_file(output),
                ..output.clone()
            },
            flips,
            set,
            config: self.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
        seed = 3
        dataset = "small-7x7"
        prototypes = [0, 2]
        variants = 3

        [[outputs]]
        flip = 0.1
        file = "out/groups-d10.npy"

        [[outputs]]
        flip = 0.0
        file = "out/groups-d00.npy"
    "#;

    fn parse(text: &str) -> Result<GroupsConfig> {
        GroupsConfig::parse(text, Path::new("../../data/experiments/x/config.toml"))
    }

    fn invalid_message(text: &str) -> String {
        match parse(text) {
            Err(Error::InvalidConfig { message, .. }) => message,
            other => panic!("expected an invalid configuration, got {other:?}"),
        }
    }

    /// `small-7x7` has `N = 49`: `round(0.1 * 49) = 5` flipped pixels give
    /// every variant overlap `(49 - 2 * 5) / 49 = 39 / 49` with its
    /// prototype. With `flip = 0` every variant equals its prototype.
    #[test]
    fn generates_groups_of_variants() {
        let config = parse(CONFIG).unwrap();
        let generated = config.generate().unwrap();
        assert_eq!(generated.len(), 2);
        let first = &generated[0];
        assert_eq!(first.flips, 5);
        assert_eq!(
            first.output.file,
            Path::new("../../data/experiments/x/out/groups-d10.npy")
        );
        assert_eq!(
            (first.set.len(), first.set.width(), first.set.height()),
            (6, 7, 7)
        );
        let source = Registry::open(None, config.directory())
            .unwrap()
            .load("small-7x7")
            .unwrap();
        for (k, variant) in first.set.iter().enumerate() {
            let prototype = source.pattern([0, 2][k / 3]);
            assert_eq!(variant.overlap(&prototype), 39.0 / 49.0, "variant {k}");
        }
        for k in 1..6 {
            assert_ne!(first.set.pattern(k), first.set.pattern(k - 1));
        }
        for (k, variant) in generated[1].set.iter().enumerate() {
            assert_eq!(variant, source.pattern([0, 2][k / 3]));
        }
        assert_eq!(config.generate().unwrap(), generated);
    }

    #[test]
    fn rejects_invalid_values() {
        let cases = [
            (
                "prototypes = [0, 2]",
                "prototypes = []",
                "`prototypes` needs",
            ),
            (
                "prototypes = [0, 2]",
                "prototypes = [2, 2]",
                "`prototypes` lists 2 twice",
            ),
            ("variants = 3", "variants = 0", "`variants` must be"),
            ("flip = 0.1", "flip = 1.5", "flip 1.5 is not a fraction"),
            ("groups-d00.npy", "groups-d10.npy", "share the file name"),
        ];
        for (from, to, expected) in cases {
            let message = invalid_message(&CONFIG.replace(from, to));
            assert!(message.contains(expected), "{to}: {message}");
        }
        let no_outputs = CONFIG.split("[[outputs]]").next().unwrap().to_owned() + "outputs = []";
        assert!(invalid_message(&no_outputs).contains("no outputs"));
        assert!(matches!(
            parse(&CONFIG.replace("variants = 3", "variants = 3\ncopies = 2")),
            Err(Error::Config { .. })
        ));
    }

    #[test]
    fn rejects_a_missing_prototype() {
        let config = parse(&CONFIG.replace("[0, 2]", "[0, 5]")).unwrap();
        let error = config.generate().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("prototype 5 does not exist, `small-7x7` has 5 patterns"),
            "{error}"
        );
    }
}
