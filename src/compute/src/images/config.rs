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
pub mod provenance;
pub mod source;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub use conversion::Conversion;
pub use output::Output;
pub use provenance::Provenance;
pub use source::Source;

use super::ImageConverter;
use crate::paths::resolve;
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
    /// valid for [`ImageConverter`], there must be at least one output, and
    /// no two outputs may share a file or a file stem (which names their
    /// preview directory). Relative paths are resolved against the parent
    /// directory of `path`.
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
        let outputs: Vec<Output> = raw
            .outputs
            .into_iter()
            .map(|output| Output {
                file: resolve(base, &output.file),
                ..output
            })
            .collect();
        let mut stems = HashSet::new();
        if let Some(output) = outputs
            .iter()
            .find(|output| !stems.insert(output.file.file_stem()))
        {
            return Err(invalid(format!(
                "two outputs share the file name of `{}`",
                output.file.display()
            )));
        }
        Ok(Self {
            path: path.to_owned(),
            converter,
            sources: resolve(base, &raw.sources),
            images: resolve(base, &raw.images),
            outputs,
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
    /// it with the method of each output. Each image must exist and match its
    /// listed SHA-1 checksum. Patterns follow the order of the source list.
    /// Nothing is written.
    pub fn convert(&self) -> Result<Vec<Conversion>> {
        let sources = self.read_sources()?;
        let grays = sources
            .iter()
            .map(|source| {
                let path = self.images.join(&source.file);
                if !path.is_file() {
                    return Err(Error::MissingImage { path });
                }
                source.verify(&path)?;
                self.converter.open(&path)
            })
            .collect::<Result<Vec<_>>>()?;

        self.outputs
            .iter()
            .map(|output| {
                let patterns: Vec<_> = grays.iter().map(|gray| output.method.apply(gray)).collect();
                Ok(Conversion {
                    output: output.clone(),
                    sources: sources.clone(),
                    config: self.path.clone(),
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
        let mut sources = String::from("file\tsha1\turl\tpage\tauthor\tlicense\tlicense_url\n");
        std::fs::create_dir(dir.join("raw")).unwrap();
        for (name, mirror) in [("a.png", false), ("b.png", true)] {
            let path = dir.join("raw").join(name);
            GrayImage::from_fn(8, 4, |x, _| {
                let x = if mirror { 7 - x } else { x };
                Luma([(x * 32) as u8])
            })
            .save(&path)
            .unwrap();
            let sha1 = Source::checksum(&path).unwrap();
            sources.push_str(&format!("{name}\t{sha1}\turl\tpage\tA\tCC0\tlicense\n"));
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
    fn rejects_outputs_sharing_a_file_name() {
        let parse = |text: &str| ConversionConfig::parse(text, Path::new("c.toml"));
        let same_file = CONFIG.replace("out/dither.npy", "out/./median.npy");
        let error = parse(&same_file).unwrap_err();
        assert!(error.to_string().contains("share the file name"), "{error}");
        let same_stem = CONFIG.replace("out/dither.npy", "other/median.npy");
        assert!(parse(&same_stem).is_err());
    }

    #[test]
    fn converts_every_image_for_every_output() {
        let dir = tempfile::tempdir().unwrap();
        let conversions = setup(dir.path()).convert().unwrap();
        assert_eq!(conversions.len(), 2);
        for conversion in &conversions {
            assert_eq!(conversion.set.len(), 2);
            assert_eq!(conversion.set.neurons(), 8);
            let files: Vec<_> = conversion.sources.iter().map(|s| s.file.as_str()).collect();
            assert_eq!(files, ["a.png", "b.png"]);
        }
        let median = &conversions[0].set;
        assert_eq!(median.pattern(0).values()[0], 1);
        assert_eq!(median.pattern(1).values()[0], -1);
    }

    #[test]
    fn reports_a_changed_image() {
        let dir = tempfile::tempdir().unwrap();
        let config = setup(dir.path());
        GrayImage::new(8, 4)
            .save(dir.path().join("raw").join("a.png"))
            .unwrap();
        let error = config.convert().unwrap_err();
        assert!(matches!(error, Error::ChecksumMismatch { .. }), "{error}");
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
