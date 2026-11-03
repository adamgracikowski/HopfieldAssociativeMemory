//! Source images of a conversion with their origin and license.

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;

use crate::{Error, Result};

/// A source image with its origin and license: one row of a source list.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Source {
    /// File name in the image directory.
    pub file: String,
    /// SHA-1 checksum of the file (hexadecimal).
    pub sha1: String,
    /// Download URL.
    pub url: String,
    /// Description page of the image.
    pub page: String,
    /// Author.
    pub author: String,
    /// License.
    pub license: String,
}

impl Source {
    /// Reads a source list: a tab-separated file with the header
    /// `file  sha1  url  page  author  license` and one image per row.
    /// The list must contain at least one image and no file name twice.
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
        if let Some(source) = sources.iter().find(|s| !seen.insert(&s.file)) {
            return Err(invalid(format!("image `{}` is listed twice", source.file)));
        }
        Ok(sources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "file\tsha1\turl\tpage\tauthor\tlicense\n";

    fn row(file: &str) -> String {
        format!("{file}\t0123\thttps://example.org/{file}\thttps://example.org/p\tA. Author\tCC0\n")
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
    fn rejects_missing_columns() {
        let error = read("file\tsha1\na.png\t00\n").unwrap_err();
        assert!(matches!(error, Error::Csv { .. }), "{error}");
    }
}
