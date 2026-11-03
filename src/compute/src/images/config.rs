//! Configuration of a photo conversion: bitmap size, list of source images
//! and output datasets.
//!
//! ```toml
//! width = 300
//! height = 200
//! sources = "sources.tsv"
//! images = "raw"
//!
//! [[outputs]]
//! method = "median"
//! file = "../../datasets/cats-300x200-median.npy"
//! ```
//!
//! Paths are relative to the directory of the configuration file. The source
//! list format is described in [`Source::read_list`]; its row order is the
//! pattern order in the output datasets.

pub mod conversion;
pub mod output;
pub mod source;

use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

pub use conversion::Conversion;
pub use output::Output;
pub use source::Source;

use super::ImageConverter;
use crate::patterns::PatternSet;
use crate::{Error, Result};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    width: usize,
    height: usize,
    sources: PathBuf,
    images: PathBuf,
    outputs: Vec<Output>,
}

/// A validated conversion configuration with paths resolved against the
/// directory of the configuration file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionConfig {
    path: PathBuf,
    converter: ImageConverter,
    sources: PathBuf,
    images: PathBuf,
    outputs: Vec<Output>,
}

impl ConversionConfig {
    /// Reads and validates the configuration file at `path`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parses and validates configuration `text`: the bitmap size must be
    /// valid for [`ImageConverter`] and there must be at least one output.
    /// Relative paths are resolved against the parent directory of `path`.
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let raw: RawConfig = toml::from_str(text).map_err(|source| Error::Config {
            path: path.to_owned(),
            source: Box::new(source),
        })?;
        let invalid = |message: String| Error::InvalidConfig {
            path: path.to_owned(),
            message,
        };
        let converter = ImageConverter::new(raw.width, raw.height)
            .map_err(|error| invalid(error.to_string()))?;
        if raw.outputs.is_empty() {
            return Err(invalid("no outputs".to_owned()));
        }
        let base = path.parent().unwrap_or(Path::new(""));
        Ok(Self {
            path: path.to_owned(),
            converter,
            sources: resolve(base, &raw.sources),
            images: resolve(base, &raw.images),
            outputs: raw
                .outputs
                .into_iter()
                .map(|output| Output {
                    file: resolve(base, &output.file),
                    ..output
                })
                .collect(),
        })
    }

    /// Configuration file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Converter to the configured bitmap size.
    pub fn converter(&self) -> ImageConverter {
        self.converter
    }

    /// Source list file.
    pub fn sources_file(&self) -> &Path {
        &self.sources
    }

    /// Directory with the source images.
    pub fn images_dir(&self) -> &Path {
        &self.images
    }

    /// Output datasets with resolved paths.
    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }

    /// Reads the source list (see [`Source::read_list`]).
    pub fn read_sources(&self) -> Result<Vec<Source>> {
        Source::read_list(&self.sources)
    }

    /// Converts every source image once to a grayscale bitmap and binarizes
    /// it with the method of each output. Patterns follow the order of the
    /// source list. Nothing is written.
    pub fn convert(&self) -> Result<Vec<Conversion>> {
        let sources = self.read_sources()?;
        let grays = sources
            .iter()
            .map(|source| {
                let path = self.images.join(&source.file);
                if !path.is_file() {
                    return Err(Error::MissingImage { path });
                }
                self.converter.open(&path)
            })
            .collect::<Result<Vec<_>>>()?;
        let names: Vec<String> = sources.into_iter().map(|source| source.file).collect();

        self.outputs
            .iter()
            .map(|output| {
                let patterns: Vec<_> = grays.iter().map(|gray| output.method.apply(gray)).collect();
                Ok(Conversion {
                    output: output.clone(),
                    names: names.clone(),
                    set: PatternSet::from_patterns(
                        self.converter.width(),
                        self.converter.height(),
                        &patterns,
                    )?,
                })
            })
            .collect()
    }
}

/// Joins `relative` to `base` and removes `.` components and `..` components
/// that follow a directory name, without touching the file system (so a `..`
/// after a symbolic link is resolved lexically).
fn resolve(base: &Path, relative: &Path) -> PathBuf {
    let mut path = PathBuf::new();
    for component in base.join(relative).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir
                if matches!(path.components().next_back(), Some(Component::Normal(_))) =>
            {
                path.pop();
            }
            other => path.push(other),
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::Binarization;
    use image::{GrayImage, Luma};

    const CONFIG: &str = r#"
        width = 4
        height = 2
        sources = "sources.tsv"
        images = "raw"

        [[outputs]]
        method = "median"
        file = "out/median.npy"

        [[outputs]]
        method = "floyd-steinberg"
        file = "out/dither.npy"
    "#;

    /// Writes the configuration, a source list and two 8x4 images: a
    /// horizontal gradient and its mirror.
    fn setup(dir: &Path) -> ConversionConfig {
        std::fs::write(dir.join("config.toml"), CONFIG).unwrap();
        let mut sources = String::from("file\tsha1\turl\tpage\tauthor\tlicense\n");
        std::fs::create_dir(dir.join("raw")).unwrap();
        for (name, mirror) in [("a.png", false), ("b.png", true)] {
            sources.push_str(&format!(
                "{name}\t00\thttps://example.org\thttps://example.org\tA\tCC0\n"
            ));
            GrayImage::from_fn(8, 4, |x, _| {
                let x = if mirror { 7 - x } else { x };
                Luma([(x * 32) as u8])
            })
            .save(dir.join("raw").join(name))
            .unwrap();
        }
        std::fs::write(dir.join("sources.tsv"), sources).unwrap();
        ConversionConfig::from_file(&dir.join("config.toml")).unwrap()
    }

    #[test]
    fn resolves_paths_against_the_config_directory() {
        let config = ConversionConfig::parse(CONFIG, Path::new("data/cats/config.toml")).unwrap();
        assert_eq!(config.path(), Path::new("data/cats/config.toml"));
        assert_eq!(config.sources_file(), Path::new("data/cats/sources.tsv"));
        assert_eq!(config.images_dir(), Path::new("data/cats/raw"));
        assert_eq!(config.outputs()[1].method, Binarization::FloydSteinberg);
        assert_eq!(
            config.outputs()[1].file,
            Path::new("data/cats/out/dither.npy")
        );
        assert_eq!(config.converter().width(), 4);
    }

    #[test]
    fn resolves_parent_directories_lexically() {
        let base = Path::new("data/images/cats");
        assert_eq!(
            resolve(base, Path::new("../../datasets/./a.npy")),
            Path::new("data/datasets/a.npy")
        );
        assert_eq!(
            resolve(Path::new("a"), Path::new("../../b.npy")),
            Path::new("../b.npy")
        );
        assert_eq!(resolve(Path::new(""), Path::new("raw")), Path::new("raw"));
    }

    #[test]
    fn rejects_invalid_configurations() {
        let parse = |text: &str| ConversionConfig::parse(text, Path::new("c.toml"));
        let error = parse(&format!("{CONFIG}\nseed = 1")).unwrap_err();
        assert!(matches!(error, Error::Config { .. }), "{error}");
        let error = parse(&CONFIG.replace("width = 4", "width = 0")).unwrap_err();
        assert!(error.to_string().contains("is empty"), "{error}");
        let no_outputs = "width = 1\nheight = 1\nsources = \"s\"\nimages = \"i\"\noutputs = []";
        assert!(
            parse(no_outputs)
                .unwrap_err()
                .to_string()
                .contains("no outputs")
        );
    }

    #[test]
    fn converts_every_image_for_every_output() {
        let dir = tempfile::tempdir().unwrap();
        let conversions = setup(dir.path()).convert().unwrap();
        assert_eq!(conversions.len(), 2);
        for conversion in &conversions {
            assert_eq!(conversion.set.len(), 2);
            assert_eq!(conversion.set.neurons(), 8);
            assert_eq!(conversion.names, ["a.png", "b.png"]);
        }
        let median = &conversions[0].set;
        assert_eq!(median.pattern(0).values()[0], 1);
        assert_eq!(median.pattern(1).values()[0], -1);
    }

    #[test]
    fn reports_a_missing_image() {
        let dir = tempfile::tempdir().unwrap();
        let config = setup(dir.path());
        std::fs::remove_file(dir.path().join("raw").join("b.png")).unwrap();
        let error = config.convert().unwrap_err();
        assert!(matches!(error, Error::MissingImage { .. }), "{error}");
    }
}
