//! Bipolar vectors shown as rectangular bitmaps.

use std::fmt;

use image::GrayImage;
use rand::{Rng, RngExt};

use super::{Pattern, PatternView};
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

    /// Copy with a band of `rows` consecutive full-width rows set to `-1`
    /// (the background), as if covered. The first row of the band is drawn
    /// uniformly from the `height - rows + 1` possible positions, so `rows =
    /// 0` leaves the bitmap unchanged and `rows = height` covers all of it.
    /// One number is drawn from `rng` in every case.
    pub fn occluded<R: Rng + ?Sized>(&self, rows: usize, rng: &mut R) -> Result<Pattern> {
        if rows > self.height {
            return Err(Error::InvalidArgument {
                message: format!(
                    "cannot cover {rows} rows of a bitmap with {} rows",
                    self.height
                ),
            });
        }
        let first = rng.random_range(0..=self.height - rows);
        let mut pixels = self.pixels.values().to_owned();
        pixels
            .iter_mut()
            .skip(first * self.width)
            .take(rows * self.width)
            .for_each(|pixel| *pixel = -1);
        Pattern::new(pixels)
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

    /// A 2 x 4 bitmap of `1`: covering 2 rows leaves 3 possible bands
    /// (rows 0-1, 1-2, 2-3); each covers exactly 4 pixels in consecutive
    /// rows, and enough seeds reach every position.
    #[test]
    fn covers_a_band_of_rows() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use std::collections::BTreeSet;

        let pattern = Pattern::new(ndarray::Array1::from_elem(8, 1)).unwrap();
        let bitmap = Bitmap::new(pattern.view(), 2, 4).unwrap();
        let mut firsts = BTreeSet::new();
        for seed in 0..50 {
            let covered = bitmap
                .occluded(2, &mut ChaCha8Rng::seed_from_u64(seed))
                .unwrap();
            let rows: Vec<usize> = (0..4)
                .filter(|&row| covered.values()[2 * row] == -1)
                .collect();
            assert_eq!(rows.len(), 2, "{covered:?}");
            assert_eq!(rows[1], rows[0] + 1);
            for row in 0..4 {
                let line: Vec<i8> = covered
                    .values()
                    .iter()
                    .skip(2 * row)
                    .take(2)
                    .copied()
                    .collect();
                let value = if rows.contains(&row) { -1 } else { 1 };
                assert_eq!(line, [value, value], "row {row}");
            }
            firsts.insert(rows[0]);
        }
        assert_eq!(firsts, BTreeSet::from([0, 1, 2]));
    }

    #[test]
    fn covers_nothing_or_everything() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let pattern = Pattern::new(array![1, -1, -1, -1, 1, 1]).unwrap();
        let bitmap = Bitmap::new(pattern.view(), 3, 2).unwrap();
        let rng = &mut ChaCha8Rng::seed_from_u64(1);
        assert_eq!(bitmap.occluded(0, rng).unwrap(), pattern);
        assert_eq!(bitmap.occluded(2, rng).unwrap().fraction_on(), 0.0);
        let error = bitmap.occluded(3, rng).unwrap_err();
        assert!(error.to_string().contains("cover 3 rows"), "{error}");
    }

    #[test]
    fn rejects_a_size_mismatch() {
        let pattern = Pattern::new(array![1, -1, 1]).unwrap();
        assert!(Bitmap::new(pattern.view(), 2, 2).is_err());
        assert!(Bitmap::new(pattern.view(), usize::MAX, 2).is_err());
    }
}
