//! Conversion of photographs to bipolar bitmaps.

pub mod binarization;
pub mod config;

use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, ImageDecoder, ImageReader};

pub use binarization::Binarization;
pub use config::{Conversion, ConversionConfig};

use crate::patterns::{Pattern, bitmap_size};
use crate::{Error, Result};

/// Converts photographs to grayscale bitmaps of a fixed size and binarizes
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageConverter {
    width: u32,
    height: u32,
}

impl ImageConverter {
    /// Creates a converter to `width x height` bitmaps after checking that
    /// the size is non-empty and fits the image library.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(invalid(format!("bitmap size {width}x{height} is empty")));
        }
        bitmap_size(width, height)?;
        let to_u32 = |n: usize| {
            u32::try_from(n).map_err(|_| invalid(format!("bitmap side {n} is too large")))
        };
        Ok(Self {
            width: to_u32(width)?,
            height: to_u32(height)?,
        })
    }

    /// Bitmap width in pixels.
    pub fn width(&self) -> usize {
        self.width as usize
    }

    /// Bitmap height in pixels.
    pub fn height(&self) -> usize {
        self.height as usize
    }

    /// Reads the image at `path`, rotates it according to its EXIF
    /// orientation and returns its [grayscale bitmap](Self::grayscale).
    pub fn open(&self, path: &Path) -> Result<GrayImage> {
        let io_error = |source| Error::Io {
            path: path.to_owned(),
            source,
        };
        let image_error = |source| Error::Image {
            path: path.to_owned(),
            source: Box::new(source),
        };
        let mut decoder = ImageReader::open(path)
            .map_err(io_error)?
            .with_guessed_format()
            .map_err(io_error)?
            .into_decoder()
            .map_err(image_error)?;
        let orientation = decoder.orientation().map_err(image_error)?;
        let mut image = DynamicImage::from_decoder(decoder).map_err(image_error)?;
        image.apply_orientation(orientation);
        Ok(self.grayscale(&image))
    }

    /// Grayscale, centre-cropped and Lanczos-scaled to `width x height`.
    pub fn grayscale(&self, image: &DynamicImage) -> GrayImage {
        DynamicImage::ImageLuma8(image.to_luma8())
            .resize_to_fill(self.width, self.height, FilterType::Lanczos3)
            .into_luma8()
    }

    /// Converts `image` to a bipolar pattern of `width * height` pixels
    /// stored row by row.
    pub fn convert(&self, image: &DynamicImage, method: Binarization) -> Pattern {
        method.apply(&self.grayscale(image))
    }

    /// Reads the image at `path` and converts it to a bipolar pattern.
    pub fn convert_file(&self, path: &Path, method: Binarization) -> Result<Pattern> {
        Ok(method.apply(&self.open(path)?))
    }
}

fn invalid(message: String) -> Error {
    Error::InvalidArgument { message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;
    use image::{ImageEncoder, Luma, Rgb, RgbImage};

    /// A `30 x 10` image with black left and right thirds and a white middle.
    fn three_stripes() -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(30, 10, |x, _| {
            if (10..20).contains(&x) {
                Rgb([255, 255, 255])
            } else {
                Rgb([0, 0, 0])
            }
        }))
    }

    #[test]
    fn scales_to_the_requested_size() {
        let converter = ImageConverter::new(6, 4).unwrap();
        assert_eq!((converter.width(), converter.height()), (6, 4));
        let gray = converter.grayscale(&three_stripes());
        assert_eq!(gray.dimensions(), (6, 4));
        let pattern = converter.convert(&three_stripes(), Binarization::FloydSteinberg);
        assert_eq!(pattern.len(), 24);
    }

    #[test]
    fn crops_the_centre_when_aspect_ratios_differ() {
        let converter = ImageConverter::new(5, 5).unwrap();
        let pattern = converter.convert(&three_stripes(), Binarization::FloydSteinberg);
        assert!(pattern.values().iter().all(|&v| v == -1), "{pattern:?}");
    }

    #[test]
    fn reads_an_image_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stripes.png");
        three_stripes().save(&path).unwrap();
        let converter = ImageConverter::new(30, 10).unwrap();
        let pattern = converter
            .convert_file(&path, Binarization::FloydSteinberg)
            .unwrap();
        assert_eq!(pattern.values()[0], 1);
        assert_eq!(pattern.values()[15], -1);
    }

    /// Little-endian TIFF block with a single EXIF tag: orientation 6, which
    /// means the stored image must be rotated 90 degrees clockwise for display.
    const ORIENTATION_6: [u8; 26] = [
        b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// A stored 16x8 JPEG with a black left half and orientation 6 is shown as
    /// an 8x16 image with a black top half. Without the rotation, scaling to
    /// 8x16 would crop the centre and leave half-black rows.
    #[test]
    fn applies_the_exif_orientation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rotated.jpg");
        let stored = GrayImage::from_fn(16, 8, |x, _| Luma([if x < 8 { 0 } else { 255 }]));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut encoder = JpegEncoder::new_with_quality(&mut file, 100);
        encoder.set_exif_metadata(ORIENTATION_6.to_vec()).unwrap();
        stored.write_with_encoder(encoder).unwrap();
        drop(file);

        let converter = ImageConverter::new(8, 16).unwrap();
        assert_eq!(converter.open(&path).unwrap().dimensions(), (8, 16));
        let pattern = converter.convert_file(&path, Binarization::Median).unwrap();
        let values = pattern.values();
        assert!(values.iter().take(8).all(|&v| v == 1), "{pattern:?}");
        assert!(values.iter().rev().take(8).all(|&v| v == -1), "{pattern:?}");
    }

    #[test]
    fn reports_missing_and_invalid_files() {
        let dir = tempfile::tempdir().unwrap();
        let converter = ImageConverter::new(2, 2).unwrap();
        let missing = converter.open(&dir.path().join("missing.png"));
        assert!(matches!(missing, Err(Error::Io { .. })));

        let text = dir.path().join("not-an-image.png");
        std::fs::write(&text, "hello").unwrap();
        let error = converter.open(&text).unwrap_err();
        assert!(matches!(error, Error::Image { .. }), "{error}");
    }

    #[test]
    fn rejects_invalid_sizes() {
        assert!(ImageConverter::new(0, 3).is_err());
        assert!(ImageConverter::new(usize::MAX, 2).is_err());
        assert!(ImageConverter::new(1 << 33, 1).is_err());
    }
}
