//! Bipolar vectors shown as rectangular bitmaps.

use std::fmt;

use super::PatternView;
use crate::{Error, Result};

/// A bipolar vector of `width * height` pixels shown as a bitmap, row by row.
///
/// [`Display`](fmt::Display) prints one line per bitmap row, `#` for `1` and
/// `.` for `-1`.
#[derive(Debug, Clone, Copy)]
pub struct Bitmap<'a> {
    pixels: PatternView<'a>,
    width: usize,
    height: usize,
}

impl<'a> Bitmap<'a> {
    /// Shows `pixels` as a `width x height` bitmap after checking that the
    /// sizes agree.
    pub fn new(pixels: PatternView<'a>, width: usize, height: usize) -> Result<Self> {
        if pixels.len() != width * height {
            return Err(Error::InvalidArgument {
                message: format!(
                    "a vector of {} pixels is not a {width}x{height} bitmap",
                    pixels.len()
                ),
            });
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }

    /// Bitmap width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Bitmap height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }
}

impl fmt::Display for Bitmap<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in self.pixels.values().exact_chunks(self.width) {
            let line: String = row
                .iter()
                .map(|&v| if v == 1 { '#' } else { '.' })
                .collect();
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::Pattern;
    use ndarray::array;

    #[test]
    fn prints_rows_top_to_bottom() {
        let pattern = Pattern::new(array![1, -1, -1, -1, 1, 1]).unwrap();
        let bitmap = Bitmap::new(pattern.view(), 3, 2).unwrap();
        assert_eq!(bitmap.to_string(), "#..\n.##\n");
        assert_eq!((bitmap.width(), bitmap.height()), (3, 2));
    }

    #[test]
    fn rejects_a_size_mismatch() {
        let pattern = Pattern::new(array![1, -1, 1]).unwrap();
        assert!(Bitmap::new(pattern.view(), 2, 2).is_err());
    }
}
