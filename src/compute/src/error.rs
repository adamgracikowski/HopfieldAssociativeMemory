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

    /// A CSV file could not be read as CSV records.
    #[error("invalid CSV file `{path}`")]
    #[diagnostic(code(compute::csv))]
    Csv {
        /// File being read.
        path: PathBuf,
        /// Underlying CSV error.
        #[source]
        source: csv::Error,
    },

    /// An `.npy` file has an invalid header, dtype, shape or data.
    #[error("invalid .npy file `{path}`: {message}")]
    #[diagnostic(code(compute::npy))]
    Npy {
        /// File being read.
        path: PathBuf,
        /// What is wrong.
        message: String,
    },

    /// An image file could not be decoded.
    #[error("cannot decode image `{path}`")]
    #[diagnostic(code(compute::image))]
    Image {
        /// Image file.
        path: PathBuf,
        /// Underlying image library error, boxed because it is large.
        #[source]
        source: Box<image::ImageError>,
    },

    /// A source image listed for conversion does not exist.
    #[error("image `{path}` not found")]
    #[diagnostic(
        code(compute::missing_image),
        help("download the images first, e.g. with `data/images/download.ps1`")
    )]
    MissingImage {
        /// Expected image file.
        path: PathBuf,
    },

    /// A source image differs from the listed one.
    #[error("image `{path}` has SHA-1 {actual}, expected {expected}")]
    #[diagnostic(
        code(compute::checksum_mismatch),
        help("delete the file and download it again, e.g. with `data/images/download.ps1`")
    )]
    ChecksumMismatch {
        /// Image file.
        path: PathBuf,
        /// Checksum from the source list.
        expected: String,
        /// Checksum of the file.
        actual: String,
    },

    /// A configuration file could not be parsed.
    #[error("invalid configuration `{path}`")]
    #[diagnostic(code(compute::config))]
    Config {
        /// Configuration file.
        path: PathBuf,
        /// Underlying TOML error, boxed because it is large.
        #[source]
        source: Box<toml::de::Error>,
    },

    /// A configuration file has an invalid value.
    #[error("invalid configuration `{path}`: {message}")]
    #[diagnostic(code(compute::invalid_config))]
    InvalidConfig {
        /// Configuration file.
        path: PathBuf,
        /// What is wrong.
        message: String,
    },

    /// A pattern file has invalid content at a known position.
    #[error("{path}:{line}:{column}: {message}")]
    #[diagnostic(code(compute::parse))]
    Parse {
        /// File being parsed.
        path: PathBuf,
        /// Line number, starting at 1.
        line: u64,
        /// Field number in the record, starting at 1.
        column: usize,
        /// What is wrong.
        message: String,
    },

    /// A pattern set does not have the expected size or values.
    #[error("invalid pattern set: {message}")]
    #[diagnostic(code(compute::shape))]
    Shape {
        /// What is wrong.
        message: String,
    },

    /// An iterative learning rule produced non-finite weights.
    #[error("{rule} diverged after {epochs} epochs")]
    #[diagnostic(code(compute::learning_diverged), help("decrease the learning rate"))]
    LearningDiverged {
        /// Name of the rule.
        rule: String,
        /// Epoch in which the weights stopped being finite.
        epochs: usize,
    },

    /// A function received an argument outside its valid range.
    #[error("invalid argument: {message}")]
    #[diagnostic(code(compute::invalid_argument))]
    InvalidArgument {
        /// What is wrong.
        message: String,
    },

    /// The dataset registry could not be parsed.
    #[error("invalid dataset registry `{path}`")]
    #[diagnostic(code(compute::registry))]
    Registry {
        /// Registry file.
        path: PathBuf,
        /// Underlying TOML error, boxed because it is large.
        #[source]
        source: Box<toml::de::Error>,
    },

    /// A registry entry has an invalid value.
    #[error("invalid dataset `{name}` in `{registry}`: {message}")]
    #[diagnostic(code(compute::invalid_entry))]
    InvalidEntry {
        /// Registry file.
        registry: PathBuf,
        /// Dataset name.
        name: String,
        /// What is wrong.
        message: String,
    },

    /// No dataset registry was found in the directory or its ancestors.
    #[error("dataset registry `{relative}` not found in `{start}` or its parent directories")]
    #[diagnostic(
        code(compute::registry_not_found),
        help("run the command inside the repository or pass the registry path explicitly")
    )]
    RegistryNotFound {
        /// Directory where the search started.
        start: PathBuf,
        /// Registry path searched for relative to each directory.
        relative: PathBuf,
    },

    /// The registry has no dataset with the requested name.
    #[error("unknown dataset `{name}` in `{registry}`")]
    #[diagnostic(code(compute::unknown_dataset), help("known datasets: {known}"))]
    UnknownDataset {
        /// Requested name.
        name: String,
        /// Registry file.
        registry: PathBuf,
        /// Comma-separated names of the registered datasets.
        known: String,
    },

    /// A pattern file has an extension the library cannot read.
    #[error("unsupported pattern file format `{path}`")]
    #[diagnostic(
        code(compute::unsupported_format),
        help("supported formats: .csv, .npy")
    )]
    UnsupportedFormat {
        /// Pattern file.
        path: PathBuf,
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

    #[test]
    fn parse_error_shows_the_position() {
        let error = Error::Parse {
            path: PathBuf::from("a.csv"),
            line: 3,
            column: 7,
            message: "expected 1 or -1, found `0`".to_owned(),
        };
        assert_eq!(error.to_string(), "a.csv:3:7: expected 1 or -1, found `0`");
    }
}
