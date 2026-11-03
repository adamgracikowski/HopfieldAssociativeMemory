# Source images

Photographs converted to large bipolar bitmaps by `compute convert-images`. Each subdirectory holds one conversion:

- `config.toml`: bitmap size, binarization methods and output datasets,
- `sources.tsv`: the images in pattern order, with SHA-1 checksum, download URL, description page, author and license,
- `raw/`: the downloaded images (not committed).

## `cats`

Twelve cat photos from [Wikimedia Commons](https://commons.wikimedia.org/) (category "Quality images of cats"), converted to 300x200 bitmaps ($N = 60000$) with two binarizations:

- `data/datasets/cats-300x200-median.npy`: threshold at the median brightness,
- `data/datasets/cats-300x200-dither.npy`: Floyd-Steinberg dithering.

To reproduce the datasets:

```powershell
pwsh -File data/images/download.ps1 -Sources data/images/cats/sources.tsv -OutDir data/images/cats/raw
cd src/compute
cargo run --release -- convert-images --config ../../data/images/cats/config.toml
```

## Licenses

The photos are licensed by their authors under the Creative Commons licenses listed in `sources.tsv` (CC BY and CC BY-SA, versions 2.0 to 4.0). Each bitmap in the converted datasets is an adaptation (cropped, scaled, converted to grayscale and binarized) of the photo in the same row of `sources.tsv` and is shared under that photo's license.
