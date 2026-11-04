//! Source images of a conversion with their origin and license.

use std::collections::HashSet;
use std::io::Read;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// A source image with its origin and license: one row of a source list.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Source {
    /// Plain file name in the image directory.
    pub file: String,
    /// SHA-1 checksum of the file (40 hexadecimal digits).
    pub sha1: String,
    /// Download URL.
    pub url: String,
    /// Description page of the image.
    pub page: String,
    /// Author, or the attribution text the author requires.
    pub author: String,
    /// License name.
    pub license: String,
    /// License text URL.
    pub license_url: String,
}

impl Source {
    /// Reads a source list: a tab-separated file with the header
    /// `file  sha1  url  page  author  license  license_url` and one image
    /// per row. The list must contain at least one image, every `file` must
    /// be a plain file name (no directories), listed once, and every `sha1`
    /// must have 40 hexadecimal digits.
    pub fn read_list(path: &Path) -> Result<Vec<Self>> {
        let csv_error = |source| Error::Csv {
            path: path.to_owned(),
            source,
        };
        let sources = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .trim(csv::Trim::All)
            .from_path(path)
            .map_err(csv_error)?
            .deserialize()
            .collect::<std::result::Result<Vec<Self>, _>>()
            .map_err(csv_error)?;

        let invalid = |message: String| Error::InvalidConfig {
            path: path.to_owned(),
            message,
        };
        if sources.is_empty() {
            return Err(invalid("no source images".to_owned()));
        }
        let mut seen = HashSet::new();
        for source in &sources {
            if !is_plain_file_name(&source.file) {
                return Err(invalid(format!(
                    "image `{}` is not a plain file name",
                    source.file
                )));
            }
            if source.sha1.len() != 40 || !source.sha1.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(invalid(format!(
                    "image `{}` has an invalid SHA-1 `{}`",
                    source.file, source.sha1
                )));
            }
            if !seen.insert(&source.file) {
                return Err(invalid(format!("image `{}` is listed twice", source.file)));
            }
        }
        Ok(sources)
    }

    /// SHA-1 checksum of the file at `path` as 40 lowercase hexadecimal
    /// digits, computed while reading the file in chunks.
    pub fn checksum(path: &Path) -> Result<String> {
        let io_error = |source| Error::Io {
            path: path.to_owned(),
            source,
        };
        let mut file = std::fs::File::open(path).map_err(io_error)?;
        let mut hasher = sha1_smol::Sha1::new();
        let mut buffer = vec![0; 1 << 16];
        loop {
            let read = file.read(&mut buffer).map_err(io_error)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(hasher.digest().to_string())
    }

    /// Checks that the file at `path` has the listed checksum (compared
    /// without regard to letter case).
    pub fn verify(&self, path: &Path) -> Result<()> {
        let actual = Self::checksum(path)?;
        if actual.eq_ignore_ascii_case(&self.sha1) {
            Ok(())
        } else {
            Err(Error::ChecksumMismatch {
                path: path.to_owned(),
                expected: self.sha1.clone(),
                actual,
            })
        }
    }
}

/// Whether `name` is a single normal path component, so that joining it to a
/// directory stays inside that directory.
fn is_plain_file_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(part)), None) if part == name
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "file\tsha1\turl\tpage\tauthor\tlicense\tlicense_url\n";
    const SHA1: &str = "da39a3ee5e6b4b0d3255bfef95601890afd80709";

    fn row(file: &str) -> String {
        format!(
            "{file}\t{SHA1}\thttps://example.org/{file}\thttps://example.org/p\tA. Author\tCC0\thttps://example.org/l\n"
        )
    }

    fn read(content: &str) -> Result<Vec<Source>> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sources.tsv");
        std::fs::write(&path, content).unwrap();
        Source::read_list(&path)
    }

    #[test]
    fn reads_rows_in_order() {
        let sources = read(&format!("{HEADER}{}{}", row("a.png"), row("b.png"))).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[1].file, "b.png");
        assert_eq!(sources[0].author, "A. Author");
        assert_eq!(sources[0].license, "CC0");
        assert_eq!(sources[0].license_url, "https://example.org/l");
    }

    #[test]
    fn rejects_an_empty_list() {
        let error = read(HEADER).unwrap_err();
        assert!(error.to_string().contains("no source images"), "{error}");
    }

    #[test]
    fn rejects_a_file_listed_twice() {
        let error = read(&format!("{HEADER}{}{}", row("a.png"), row("a.png"))).unwrap_err();
        assert!(error.to_string().contains("listed twice"), "{error}");
    }

    #[test]
    fn rejects_file_names_with_directories() {
        for name in ["../a.png", "raw/a.png", "/a.png", ".", ".."] {
            let error = read(&format!("{HEADER}{}", row(name))).unwrap_err();
            assert!(
                error.to_string().contains("not a plain file name"),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn rejects_an_invalid_checksum() {
        let error = read(&format!("{HEADER}{}", row("a.png").replace(SHA1, "abc"))).unwrap_err();
        assert!(error.to_string().contains("invalid SHA-1"), "{error}");
    }

    #[test]
    fn rejects_missing_columns() {
        let error = read("file\tsha1\na.png\t00\n").unwrap_err();
        assert!(matches!(error, Error::Csv { .. }), "{error}");
    }

    #[test]
    fn verifies_the_checksum_of_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.png");
        std::fs::write(&path, "").unwrap();
        assert_eq!(Source::checksum(&path).unwrap(), SHA1);

        let mut source = read(&format!("{HEADER}{}", row("a.png")))
            .unwrap()
            .remove(0);
        source.sha1 = SHA1.to_uppercase();
        source.verify(&path).unwrap();

        std::fs::write(&path, "changed").unwrap();
        let error = source.verify(&path).unwrap_err();
        assert!(matches!(error, Error::ChecksumMismatch { .. }), "{error}");
    }
}
