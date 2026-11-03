//! Pattern sets stored as CSV: one pattern per line, pixels `1` or `-1`.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use ::csv::{ReaderBuilder, StringRecord, Trim};
use ndarray::Array2;

use super::{PatternSet, bitmap_size};
use crate::{Error, Result};

/// Reads a CSV pattern set of `width x height` bitmaps with fields separated
/// by `separator`.
///
/// The file has no header. Lines that are empty or contain only whitespace
/// are skipped, whitespace around values and a UTF-8 byte order mark are
/// ignored.
pub fn read_csv(path: &Path, width: usize, height: usize, separator: u8) -> Result<PatternSet> {
    let file = std::fs::File::open(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    parse_csv(file, path, width, height, separator)
}

/// Parses CSV data of a pattern set. `path` is used only in error messages.
///
/// The input is read line by line (`\n` or `\r\n`) and every line is split
/// into fields by the `csv` crate. Counting lines here, not in the `csv`
/// reader, keeps the line numbers in errors equal to the lines of the file,
/// also after skipped lines and in CRLF files.
pub fn parse_csv(
    reader: impl Read,
    path: &Path,
    width: usize,
    height: usize,
    separator: u8,
) -> Result<PatternSet> {
    let neurons = bitmap_size(width, height)?;
    let parse_error = |line: u64, column: usize, message: String| Error::Parse {
        path: path.to_owned(),
        line,
        column,
        message,
    };

    let mut values = Vec::new();
    let mut rows = 0;
    for (index, line) in BufReader::new(reader).lines().enumerate() {
        let line_number = index as u64 + 1;
        let line = line.map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        let content = match index {
            0 => line.strip_prefix('\u{feff}').unwrap_or(&line),
            _ => &line,
        };
        if content.trim().is_empty() {
            continue;
        }
        let record = split_line(content, separator).map_err(|source| Error::Csv {
            path: path.to_owned(),
            source,
        })?;
        for (column, field) in record.iter().enumerate() {
            let value = match field {
                "1" => 1,
                "-1" => -1,
                other => {
                    return Err(parse_error(
                        line_number,
                        column + 1,
                        format!("expected 1 or -1, found `{other}`"),
                    ));
                }
            };
            values.push(value);
        }
        if record.len() != neurons {
            return Err(parse_error(
                line_number,
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
        .expect("every line was checked to have `neurons` values");
    PatternSet::new(width, height, data)
}

/// Fields of one non-empty line, trimmed.
fn split_line(line: &str, separator: u8) -> ::csv::Result<StringRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(separator)
        .trim(Trim::All)
        .flexible(true)
        .from_reader(line.as_bytes());
    let mut record = StringRecord::new();
    reader.read_record(&mut record)?;
    Ok(record)
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

    #[test]
    fn reports_the_line_after_blank_lines() {
        let error = parse("1,-1\n\n\n1,0\n", 2, 1).unwrap_err();
        assert_eq!(
            error.to_string(),
            "test.csv:4:2: expected 1 or -1, found `0`"
        );
    }

    #[test]
    fn reports_the_line_in_crlf_files() {
        let error = parse("1,-1\r\n1,1\r\n-1,1\r\n-1,-1\r\n1,0\r\n", 2, 1).unwrap_err();
        assert_eq!(
            error.to_string(),
            "test.csv:5:2: expected 1 or -1, found `0`"
        );
    }

    #[test]
    fn skips_whitespace_only_lines() {
        let set = parse("1,-1\n   \n\t\n-1,1\n", 2, 1).unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn ignores_a_byte_order_mark() {
        let set = parse("\u{feff}1,-1\n", 2, 1).unwrap();
        assert_eq!(set.patterns(), array![[1, -1]]);
    }
}
