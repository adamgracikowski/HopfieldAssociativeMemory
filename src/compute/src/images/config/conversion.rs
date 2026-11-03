//! Pattern sets produced by a conversion and their files.

use std::path::Path;

use super::Output;
use crate::patterns::PatternSet;
use crate::patterns::npy::write_npy;
use crate::{Error, Result};

/// Pattern set produced for one output of a conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion {
    /// Output the set belongs to.
    pub output: Output,
    /// Source file names, one per pattern.
    pub names: Vec<String>,
    /// Converted patterns.
    pub set: PatternSet,
}

impl Conversion {
    /// Writes the set to the output `.npy` file, creating its directory.
    pub fn write(&self) -> Result<()> {
        let file = &self.output.file;
        if let Some(parent) = file.parent().filter(|p| !p.as_os_str().is_empty()) {
            create_dir(parent)?;
        }
        write_npy(&self.set, file)
    }

    /// Writes every pattern as a black-and-white PNG (black for `1`) to
    /// `dir/<output file stem>/<index>-<source file stem>.png`.
    pub fn write_preview(&self, dir: &Path) -> Result<()> {
        let dir = dir.join(self.output.file.file_stem().unwrap_or_default());
        create_dir(&dir)?;
        for (mu, name) in self.names.iter().enumerate() {
            let source_stem = Path::new(name).file_stem().unwrap_or_default();
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

    fn conversion(file: &Path) -> Conversion {
        Conversion {
            output: Output {
                method: Binarization::Median,
                file: file.to_owned(),
            },
            names: vec!["a.jpg".to_owned(), "b.png".to_owned()],
            set: PatternSet::new(3, 1, array![[1, -1, 1], [-1, -1, 1]]).unwrap(),
        }
    }

    #[test]
    fn writes_the_set_creating_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let conversion = conversion(&dir.path().join("out").join("set.npy"));
        conversion.write().unwrap();
        assert_eq!(read_npy(&conversion.output.file).unwrap(), conversion.set);
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
