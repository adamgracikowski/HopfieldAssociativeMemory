//! Binarization of grayscale images to bipolar patterns.

use image::GrayImage;
use image::imageops::{BiLevel, ColorMap, dither};
use ndarray::Array1;
use serde::Deserialize;

use crate::patterns::{BipolarVector, Pattern};

/// Method that turns grayscale pixels into the values `1` (dark) and `-1`
/// (light).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Binarization {
    /// Threshold at the median brightness.
    ///
    /// The threshold `m` is the smallest brightness such that more than half
    /// of the pixels have brightness at most `m`. Pixels darker than `m`
    /// become `1`, the others `-1`. At most half of the pixels are `1`, and
    /// exactly half when no other pixel has brightness `m`. A uniform image
    /// becomes all `-1`.
    Median,
    /// Floyd-Steinberg error diffusion to black and white
    /// ([`image::imageops::dither`] with [`BiLevel`], which maps brightness
    /// above 127 to white). Black pixels become `1`, white pixels `-1`.
    /// Images narrower than 2 pixels, which the image library cannot dither,
    /// are thresholded the same way without error diffusion.
    FloydSteinberg,
}

impl Binarization {
    /// Name of the method as written in configuration files.
    pub fn name(self) -> &'static str {
        match self {
            Self::Median => "median",
            Self::FloydSteinberg => "floyd-steinberg",
        }
    }

    /// Binarizes `image` into a pattern of `width * height` pixels stored row
    /// by row, like the rows of a [`PatternSet`](crate::PatternSet).
    pub fn apply(self, image: &GrayImage) -> Pattern {
        let values: Vec<i8> = match self {
            Self::Median => {
                let threshold = median_threshold(image.as_raw());
                image
                    .as_raw()
                    .iter()
                    .map(|&luma| if luma < threshold { 1 } else { -1 })
                    .collect()
            }
            Self::FloydSteinberg => {
                let mut dithered = image.clone();
                if image.width() >= 2 {
                    dither(&mut dithered, &BiLevel);
                } else {
                    for pixel in dithered.pixels_mut() {
                        BiLevel.map_color(pixel);
                    }
                }
                dithered
                    .as_raw()
                    .iter()
                    .map(|&luma| if luma == 0 { 1 } else { -1 })
                    .collect()
            }
        };
        BipolarVector::new_unchecked(Array1::from(values))
    }
}

/// Smallest brightness `m` such that more than half of `pixels` are at most
/// `m`, found from a 256-bin histogram.
fn median_threshold(pixels: &[u8]) -> u8 {
    let mut histogram = [0usize; 256];
    for &luma in pixels {
        histogram[usize::from(luma)] += 1;
    }
    let half = pixels.len() / 2;
    let mut at_most = 0;
    for (luma, count) in histogram.iter().enumerate() {
        at_most += count;
        if at_most > half {
            return luma as u8;
        }
    }
    u8::MAX
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn gray(width: u32, height: u32, f: impl Fn(u32, u32) -> u8) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| Luma([f(x, y)]))
    }

    fn count_on(pattern: &Pattern) -> usize {
        pattern.values().iter().filter(|&&v| v == 1).count()
    }

    #[test]
    fn median_splits_distinct_values_in_half() {
        let image = gray(16, 16, |x, y| (y * 16 + x) as u8);
        let pattern = Binarization::Median.apply(&image);
        assert_eq!(pattern.len(), 256);
        assert_eq!(count_on(&pattern), 128);
        assert!(pattern.values().iter().take(128).all(|&v| v == 1));
    }

    #[test]
    fn median_puts_ties_at_the_threshold_to_minus_one() {
        let image = gray(6, 1, |x, _| [0, 0, 10, 10, 10, 20][x as usize]);
        let pattern = Binarization::Median.apply(&image);
        assert_eq!(pattern.values().to_vec(), vec![1, 1, -1, -1, -1, -1]);
    }

    #[test]
    fn median_of_a_uniform_image_is_all_minus_one() {
        let pattern = Binarization::Median.apply(&gray(5, 3, |_, _| 77));
        assert_eq!(count_on(&pattern), 0);
    }

    #[test]
    fn pixels_are_stored_row_by_row() {
        let image = gray(4, 3, |x, _| if x == 0 { 0 } else { 255 });
        for method in [Binarization::Median, Binarization::FloydSteinberg] {
            let pattern = Binarization::apply(method, &image);
            let on: Vec<usize> = (0..pattern.len())
                .filter(|&i| pattern.values()[i] == 1)
                .collect();
            assert_eq!(on, vec![0, 4, 8], "{}", method.name());
        }
    }

    #[test]
    fn dithering_keeps_black_and_white_and_halves_mid_gray() {
        let black = Binarization::FloydSteinberg.apply(&gray(8, 8, |_, _| 0));
        let white = Binarization::FloydSteinberg.apply(&gray(8, 8, |_, _| 255));
        assert_eq!((count_on(&black), count_on(&white)), (64, 0));

        let mid = Binarization::FloydSteinberg.apply(&gray(64, 64, |_, _| 128));
        let fraction = count_on(&mid) as f64 / mid.len() as f64;
        assert!((0.45..=0.55).contains(&fraction), "{fraction}");
    }

    #[test]
    fn dithering_handles_images_one_pixel_wide() {
        let image = gray(1, 4, |_, y| [0, 100, 200, 255][y as usize]);
        let pattern = Binarization::FloydSteinberg.apply(&image);
        assert_eq!(pattern.values().to_vec(), vec![1, 1, -1, -1]);
        let single = Binarization::FloydSteinberg.apply(&gray(1, 1, |_, _| 128));
        assert_eq!(single.values().to_vec(), vec![-1]);
    }

    #[test]
    fn methods_parse_from_kebab_case() {
        #[derive(Deserialize)]
        struct Wrapper {
            method: Binarization,
        }
        let parse = |s: &str| toml::from_str::<Wrapper>(&format!("method = \"{s}\""));
        assert_eq!(parse("median").unwrap().method, Binarization::Median);
        assert_eq!(
            parse("floyd-steinberg").unwrap().method,
            Binarization::FloydSteinberg
        );
        assert!(parse("otsu").is_err());
        assert_eq!(Binarization::FloydSteinberg.name(), "floyd-steinberg");
    }
}
