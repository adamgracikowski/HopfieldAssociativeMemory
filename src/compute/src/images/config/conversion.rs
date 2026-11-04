//! Pattern sets produced by a conversion and their files.

use std::path::{Path, PathBuf};

use super::{Output, Provenance, Source};
use crate::patterns::PatternSet;
use crate::patterns::npy::write_npy;
use crate::{Error, Result};

/// Pattern set produced for one output of a conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion {
    /// Output the set belongs to.
    pub output: Output,
    /// Source images, one per pattern.
    pub sources: Vec<Source>,
    /// Converted patterns.
    pub set: PatternSet,
    /// Configuration file of the conversion.
    pub config: PathBuf,
}

impl Conversion {
    /// Writes the set to the output `.npy` file, creating its directory,
    /// and its [`Provenance`] next to it, with the extension `.json`.
    pub fn write(&self) -> Result<()> {
        let file = &self.output.file;
        if let Some(parent) = file.parent().filter(|p| !p.as_os_str().is_empty()) {
            create_dir(parent)?;
        }
        write_npy(&self.set, file)?;
        let path = self.provenance_file();
        let io_error = |source: std::io::Error| Error::Io {
            path: path.clone(),
            source,
        };
        let mut text = serde_json::to_string_pretty(&Provenance::of(self))
            .map_err(|error| io_error(error.into()))?;
        text.push('\n');
        std::fs::write(&path, text).map_err(io_error)
    }

    /// File of the provenance: the output file with the extension `.json`.
    pub fn provenance_file(&self) -> PathBuf {
        self.output.file.with_extension("json")
    }

    /// Writes every pattern as a black-and-white PNG (black for `1`) to
    /// `dir/<output file stem>/<index>-<source file stem>.png`.
    pub fn write_preview(&self, dir: &Path) -> Result<()> {
        let dir = dir.join(self.output.file.file_stem().unwrap_or_default());
        create_dir(&dir)?;
        for (mu, source) in self.sources.iter().enumerate() {
            let source_stem = Path::new(&source.file).file_stem().unwrap_or_default();
            let path = dir.join(format!("{mu:02}-{}.png", source_stem.display()));
            self.set
                .bitmap(mu)
                .to_image()
                .save(&path)
                .map_err(|source| Error::Image {
                    path,
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }
}

fn create_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|source| Error::Io {
        path: dir.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::Binarization;
    use crate::patterns::npy::read_npy;
    use ndarray::array;

    fn source(file: &str) -> Source {
        Source {
            file: file.to_owned(),
            sha1: "0".repeat(40),
            url: format!("https://example.org/{file}"),
            page: "page".to_owned(),
            author: "Author".to_owned(),
            license: "CC0".to_owned(),
            license_url: "license".to_owned(),
        }
    }

    fn conversion(file: &Path) -> Conversion {
        Conversion {
            output: Output {
                method: Binarization::Median,
                file: file.to_owned(),
            },
            sources: vec![source("a.jpg"), source("b.png")],
            set: PatternSet::new(3, 1, array![[1, -1, 1], [-1, -1, 1]]).unwrap(),
            config: file.parent().unwrap().join("images").join("config.toml"),
        }
    }

    #[test]
    fn writes_the_set_and_its_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let conversion = conversion(&dir.path().join("out").join("set.npy"));
        conversion.write().unwrap();
        assert_eq!(read_npy(&conversion.output.file).unwrap(), conversion.set);

        let path = dir.path().join("out").join("set.json");
        assert_eq!(conversion.provenance_file(), path);
        let text = std::fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["config"], "images/config.toml");
        assert_eq!(json["method"], "median");
        assert_eq!(json["sources"][1]["file"], "b.png");
    }

    #[test]
    fn writes_one_preview_per_pattern() {
        let dir = tempfile::tempdir().unwrap();
        conversion(Path::new("set.npy"))
            .write_preview(dir.path())
            .unwrap();
        let first = image::open(dir.path().join("set").join("00-a.png"))
            .unwrap()
            .into_luma8();
        assert_eq!(first.dimensions(), (3, 1));
        assert_eq!(first.as_raw(), &[0, 255, 0]);
        assert!(dir.path().join("set").join("01-b.png").is_file());
    }
}
