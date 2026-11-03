//! Bipolar vectors shown as rectangular bitmaps.

use std::fmt;

use image::GrayImage;

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
        if width.checked_mul(height) != Some(pixels.len()) {
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

    /// The bitmap as a grayscale image: black for `1`, white for `-1`.
    ///
    /// # Panics
    ///
    /// Panics if a side does not fit in `u32`.
    pub fn to_image(&self) -> GrayImage {
        let side = |n: usize| u32::try_from(n).expect("bitmap side fits in u32");
        let pixels = self
            .pixels
            .values()
            .iter()
            .map(|&v| if v == 1 { 0 } else { 255 })
            .collect();
        GrayImage::from_raw(side(self.width), side(self.height), pixels)
            .expect("a bitmap has `width * height` pixels")
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
    fn converts_to_a_black_and_white_image() {
        let pattern = Pattern::new(array![1, -1, -1, -1, 1, 1]).unwrap();
        let image = Bitmap::new(pattern.view(), 3, 2).unwrap().to_image();
        assert_eq!(image.dimensions(), (3, 2));
        assert_eq!(image.as_raw(), &[0, 255, 255, 255, 0, 0]);
    }

    #[test]
    fn rejects_a_size_mismatch() {
        let pattern = Pattern::new(array![1, -1, 1]).unwrap();
        assert!(Bitmap::new(pattern.view(), 2, 2).is_err());
        assert!(Bitmap::new(pattern.view(), usize::MAX, 2).is_err());
    }
}
