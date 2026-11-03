//! Error type shared by the library.

use std::path::PathBuf;

/// Errors returned by the compute library.
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum Error {
    /// Reading or writing a file failed.
    #[error("I/O error on `{path}`")]
    #[diagnostic(code(compute::io))]
    Io {
        /// File that was being read or written.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Result type of the compute library.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_names_the_path_and_keeps_the_source() {
        let error = Error::Io {
            path: PathBuf::from("data/missing.csv"),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        assert_eq!(error.to_string(), "I/O error on `data/missing.csv`");
        assert!(std::error::Error::source(&error).is_some());
    }
}
