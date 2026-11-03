//! Pattern sets stored as NumPy `.npy` files: `int8` arrays of shape
//! `(p, height, width)` in C order.

use std::io::{BufWriter, Read, Write};
use std::path::Path;

use ndarray::Array2;
use npyz::{NpyFile, Order, WriteOptions, WriterBuilder};

use super::{PatternSet, bitmap_size};
use crate::{Error, Result};

/// Writes a pattern set to `path` as an `int8` array of shape
/// `(p, height, width)`.
pub fn write_npy(set: &PatternSet, path: &Path) -> Result<()> {
    let io_error = |source| Error::Io {
        path: path.to_owned(),
        source,
    };
    let file = std::fs::File::create(path).map_err(io_error)?;
    let mut writer = BufWriter::new(file);
    write_npy_to(set, &mut writer).map_err(io_error)?;
    writer.flush().map_err(io_error)
}

/// Writes a pattern set in the `.npy` format to `writer`.
///
/// Rows of the `p x N` matrix are row-major bitmaps, so its elements in
/// logical order are exactly the C-order elements of `(p, height, width)`
/// and are written without reshaping.
pub fn write_npy_to(set: &PatternSet, writer: impl Write) -> std::io::Result<()> {
    let shape = [set.len(), set.height(), set.width()].map(|n| n as u64);
    let mut npy = WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(writer)
        .begin_nd()?;
    npy.extend(set.patterns().iter().copied())?;
    npy.finish()
}

/// Reads a pattern set from an `.npy` file of shape `(p, height, width)`.
pub fn read_npy(path: &Path) -> Result<PatternSet> {
    let file = std::fs::File::open(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    parse_npy(std::io::BufReader::new(file), path)
}

/// Parses `.npy` data of a pattern set. `path` is used only in error
/// messages.
pub fn parse_npy(reader: impl Read, path: &Path) -> Result<PatternSet> {
    let npy_error = |message: String| Error::Npy {
        path: path.to_owned(),
        message,
    };
    let npy = NpyFile::new(reader).map_err(|e| npy_error(e.to_string()))?;

    let [p, height, width] = match npy.shape() {
        &[p, height, width] => [p, height, width].map(|n| n as usize),
        shape => {
            return Err(npy_error(format!(
                "expected shape (p, height, width), found {shape:?}"
            )));
        }
    };
    if npy.order() != Order::C {
        return Err(npy_error(
            "expected C order, found Fortran order".to_owned(),
        ));
    }
    let values = npy
        .data::<i8>()
        .map_err(|e| npy_error(e.to_string()))?
        .collect::<std::io::Result<Vec<i8>>>()
        .map_err(|e| npy_error(e.to_string()))?;

    let data = Array2::from_shape_vec((p, bitmap_size(width, height)?), values)
        .map_err(|e| npy_error(e.to_string()))?;
    PatternSet::new(width, height, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample() -> PatternSet {
        PatternSet::new(3, 2, array![[1, -1, 1, -1, -1, 1], [-1, -1, -1, 1, 1, 1]]).unwrap()
    }

    fn encode(set: &PatternSet) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_npy_to(set, &mut bytes).unwrap();
        bytes
    }

    fn parse(bytes: &[u8]) -> Result<PatternSet> {
        parse_npy(bytes, Path::new("test.npy"))
    }

    #[test]
    fn writes_int8_with_shape_p_height_width() {
        let bytes = encode(&sample());
        assert!(bytes.starts_with(b"\x93NUMPY"));
        let header = String::from_utf8_lossy(&bytes[..bytes.len() - 12]);
        assert!(header.contains("'descr': '|i1'"), "{header}");
        assert!(header.contains("'fortran_order': False"), "{header}");
        assert!(header.contains("'shape': (2, 2, 3,"), "{header}");
        assert_eq!(
            &bytes[bytes.len() - 12..],
            &[1, 255, 1, 255, 255, 1, 255, 255, 255, 1, 1, 1]
        );
    }

    #[test]
    fn round_trip_keeps_the_set() {
        assert_eq!(parse(&encode(&sample())).unwrap(), sample());
    }

    #[test]
    fn rejects_a_shape_without_three_dimensions() {
        let mut bytes = Vec::new();
        let mut npy = WriteOptions::new()
            .default_dtype()
            .shape(&[2, 3])
            .writer(&mut bytes)
            .begin_nd()
            .unwrap();
        npy.extend([1i8; 6]).unwrap();
        npy.finish().unwrap();
        let error = parse(&bytes).unwrap_err();
        assert!(error.to_string().contains("expected shape"), "{error}");
    }

    #[test]
    fn rejects_another_dtype() {
        let mut bytes = Vec::new();
        let mut npy = WriteOptions::new()
            .default_dtype()
            .shape(&[1, 1, 2])
            .writer(&mut bytes)
            .begin_nd()
            .unwrap();
        npy.extend([1.0f64, -1.0]).unwrap();
        npy.finish().unwrap();
        assert!(matches!(parse(&bytes).unwrap_err(), Error::Npy { .. }));
    }

    #[test]
    fn rejects_values_other_than_plus_minus_one() {
        let mut bytes = encode(&sample());
        let last = bytes.len() - 1;
        bytes[last] = 0;
        assert!(matches!(parse(&bytes).unwrap_err(), Error::Shape { .. }));
    }

    #[test]
    fn rejects_fortran_order() {
        let mut bytes = Vec::new();
        let mut npy = WriteOptions::new()
            .default_dtype()
            .order(Order::Fortran)
            .shape(&[1, 1, 2])
            .writer(&mut bytes)
            .begin_nd()
            .unwrap();
        npy.extend([1i8, -1]).unwrap();
        npy.finish().unwrap();
        let error = parse(&bytes).unwrap_err();
        assert!(error.to_string().contains("Fortran order"), "{error}");
    }
}
