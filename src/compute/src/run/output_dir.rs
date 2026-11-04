//! Output directory of a run.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use npyz::{AutoSerialize, WriteOptions, WriterBuilder};
use serde::Serialize;

use crate::{Error, Result};

/// Directory that receives the files of one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputDir {
    path: PathBuf,
}

impl OutputDir {
    /// Creates `path` with its parents. An existing directory must be empty
    /// unless `force` is set; then the files of the run replace files of the
    /// same name and other files stay.
    pub fn create(path: &Path, force: bool) -> Result<Self> {
        let io_error = |source| Error::Io {
            path: path.to_owned(),
            source,
        };
        if path.is_dir() && !force && std::fs::read_dir(path).map_err(io_error)?.next().is_some() {
            return Err(Error::OutputExists {
                path: path.to_owned(),
            });
        }
        std::fs::create_dir_all(path).map_err(io_error)?;
        Ok(Self {
            path: path.to_owned(),
        })
    }

    /// The directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes `value` as pretty-printed JSON to the file `name`.
    pub fn write_json<T: Serialize + ?Sized>(&self, name: &str, value: &T) -> Result<()> {
        self.write(name, |writer| {
            serde_json::to_writer_pretty(&mut *writer, value).map_err(std::io::Error::from)?;
            writer.write_all(b"\n")
        })
    }

    /// Writes `values` in C order as an `.npy` array of shape `shape` to the
    /// file `name`.
    ///
    /// # Panics
    ///
    /// Panics if the number of values differs from the product of `shape`.
    pub fn write_npy<T: AutoSerialize>(
        &self,
        name: &str,
        shape: &[usize],
        values: impl IntoIterator<Item = T>,
    ) -> Result<()> {
        let shape: Vec<u64> = shape.iter().map(|&n| n as u64).collect();
        self.write(name, |writer| {
            let mut npy = WriteOptions::new()
                .default_dtype()
                .shape(&shape)
                .writer(writer)
                .begin_nd()?;
            npy.extend(values)?;
            npy.finish()
        })
    }

    fn write(
        &self,
        name: &str,
        content: impl FnOnce(&mut BufWriter<std::fs::File>) -> std::io::Result<()>,
    ) -> Result<()> {
        let path = self.path.join(name);
        let io_error = |source| Error::Io {
            path: path.clone(),
            source,
        };
        let file = std::fs::File::create(&path).map_err(io_error)?;
        let mut writer = BufWriter::new(file);
        content(&mut writer).map_err(io_error)?;
        writer.flush().map_err(io_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use npyz::NpyFile;

    #[test]
    fn refuses_a_non_empty_directory_without_force() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("a").join("run");
        let created = OutputDir::create(&out, false).unwrap();
        assert!(out.is_dir());
        assert!(OutputDir::create(&out, false).is_ok());

        created.write_json("x.json", &[1, 2]).unwrap();
        let error = OutputDir::create(&out, false).unwrap_err();
        assert!(matches!(error, Error::OutputExists { .. }), "{error}");
        assert!(OutputDir::create(&out, true).is_ok());
    }

    #[test]
    fn writes_json_and_npy() {
        let dir = tempfile::tempdir().unwrap();
        let out = OutputDir::create(dir.path(), true).unwrap();
        out.write_json("v.json", &serde_json::json!({ "a": 1 }))
            .unwrap();
        let text = std::fs::read_to_string(dir.path().join("v.json")).unwrap();
        assert_eq!(text, "{\n  \"a\": 1\n}\n");

        out.write_npy("m.npy", &[2, 3], (0..6).map(f64::from))
            .unwrap();
        let bytes = std::fs::read(dir.path().join("m.npy")).unwrap();
        let npy = NpyFile::new(&bytes[..]).unwrap();
        assert_eq!(npy.shape(), [2, 3]);
        assert_eq!(
            npy.into_vec::<f64>().unwrap(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }
}
