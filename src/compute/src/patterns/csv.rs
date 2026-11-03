//! Pattern sets stored as CSV: one pattern per record, pixels `1` or `-1`.

use std::io::Read;
use std::path::Path;

use ::csv::{ReaderBuilder, Trim};
use ndarray::Array2;

use super::PatternSet;
use crate::{Error, Result};

/// Reads a CSV pattern set of `width x height` bitmaps with fields separated
/// by `separator`.
///
/// The file has no header. Empty lines are skipped and whitespace around
/// values is ignored.
pub fn read_csv(path: &Path, width: usize, height: usize, separator: u8) -> Result<PatternSet> {
    let file = std::fs::File::open(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    parse_csv(file, path, width, height, separator)
}

/// Parses CSV data of a pattern set. `path` is used only in error messages.
pub fn parse_csv(
    reader: impl Read,
    path: &Path,
    width: usize,
    height: usize,
    separator: u8,
) -> Result<PatternSet> {
    let neurons = width * height;
    let parse_error = |line: u64, column: usize, message: String| Error::Parse {
        path: path.to_owned(),
        line,
        column,
        message,
    };

    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(separator)
        .trim(Trim::All)
        .flexible(true)
        .from_reader(reader);

    let mut values = Vec::new();
    let mut rows = 0;
    for record in reader.records() {
        let record = record.map_err(|source| Error::Csv {
            path: path.to_owned(),
            source,
        })?;
        let line = record.position().map_or(0, |position| position.line());
        for (column, field) in record.iter().enumerate() {
            let value = match field {
                "1" => 1,
                "-1" => -1,
                other => {
                    return Err(parse_error(
                        line,
                        column + 1,
                        format!("expected 1 or -1, found `{other}`"),
                    ));
                }
            };
            values.push(value);
        }
        if record.len() != neurons {
            return Err(parse_error(
                line,
                record.len(),
                format!(
                    "pattern has {} pixels, expected {width}x{height} = {neurons}",
                    record.len()
                ),
            ));
        }
        rows += 1;
    }

    let data = Array2::from_shape_vec((rows, neurons), values)
        .expect("every record was checked to have `neurons` values");
    PatternSet::new(width, height, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn parse(text: &str, width: usize, height: usize) -> Result<PatternSet> {
        parse_csv(text.as_bytes(), Path::new("test.csv"), width, height, b',')
    }

    #[test]
    fn parses_patterns_record_by_record() {
        let set = parse("1,-1,1,1\n\n -1 , -1,1,-1\r\n", 2, 2).unwrap();
        assert_eq!(set.patterns(), array![[1, -1, 1, 1], [-1, -1, 1, -1]]);
    }

    #[test]
    fn uses_the_given_separator() {
        let set = parse_csv("1;-1\n-1;1\n".as_bytes(), Path::new("t.csv"), 2, 1, b';').unwrap();
        assert_eq!(set.patterns(), array![[1, -1], [-1, 1]]);
    }

    #[test]
    fn reports_the_position_of_an_invalid_value() {
        let error = parse("1,-1,1,1\n1,1,0,1\n", 2, 2).unwrap_err();
        assert_eq!(
            error.to_string(),
            "test.csv:2:3: expected 1 or -1, found `0`"
        );
    }

    #[test]
    fn rejects_a_record_of_the_wrong_length() {
        let error = parse("1,-1,1,1\n1,1,1\n", 2, 2).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("test.csv:2:3: pattern has 3 pixels"),
            "{error}"
        );
    }

    #[test]
    fn rejects_records_that_do_not_match_the_bitmap_size() {
        assert!(parse("1,-1,1,1\n", 3, 1).is_err());
    }

    #[test]
    fn rejects_a_file_without_patterns() {
        assert!(parse("\n\n", 2, 2).is_err());
    }
}
