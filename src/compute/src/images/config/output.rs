//! Output datasets of a conversion.

use std::path::PathBuf;

use serde::Deserialize;

use crate::images::Binarization;

/// One output dataset of a conversion.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    /// Binarization method.
    pub method: Binarization,
    /// `.npy` file to write.
    pub file: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_method_and_file() {
        let output: Output = toml::from_str("method = \"median\"\nfile = \"a.npy\"").unwrap();
        assert_eq!(output.method, Binarization::Median);
        assert_eq!(output.file, PathBuf::from("a.npy"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let text = "method = \"median\"\nfile = \"a.npy\"\nscale = 2";
        assert!(toml::from_str::<Output>(text).is_err());
    }
}
