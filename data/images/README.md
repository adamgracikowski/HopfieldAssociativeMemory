# Source images

Photographs converted to large bipolar bitmaps by `compute convert-images`. Each subdirectory holds one conversion:

- `config.toml`: bitmap size, binarization methods and output datasets,
- `sources.tsv`: the images in pattern order, with SHA-1 checksum, download URL, description page, author (or the attribution text the author requires), license and license URL,
- `raw/`: the downloaded images (not committed).

## `cats`

Twelve cat photos from [Wikimedia Commons](https://commons.wikimedia.org/) (categories "Featured pictures of cats" and "Quality images of cats"), converted to 300x200 bitmaps ($N = 60000$) with two binarizations:

- `data/datasets/cats-300x200-median.npy`: threshold at the median brightness,
- `data/datasets/cats-300x200-dither.npy`: Floyd-Steinberg dithering.

Each dataset has a provenance file of the same name with the extension `.json`, written by `convert-images`: the build of `compute` that produced it, the configuration, the binarization and the source images in pattern order.

To reproduce the datasets:

```powershell
pwsh -File data/images/download.ps1 -Sources data/images/cats/sources.tsv -OutDir data/images/cats/raw
cd src/compute
cargo run --release -- convert-images --config ../../data/images/cats/config.toml
```

Both `download.ps1` and `convert-images` check the SHA-1 checksum of every image, so a changed photo is reported instead of silently changing the datasets. The conversion is deterministic on a given machine. JPEG decoding and scaling may round differently on another CPU or library version, so the committed `.npy` files are the reference datasets.

## Licenses

The photos are licensed by their authors under the Creative Commons licenses listed in `sources.tsv` (CC BY and CC BY-SA, versions 2.0 to 4.0, with links to the license texts). When reusing a photo or bitmap, credit it with the `author` column, the license and a link to the description page. Each bitmap in the converted datasets is an adaptation (cropped, scaled, converted to grayscale and binarized) of the photo in the same row of `sources.tsv` and is shared under that photo's license.
