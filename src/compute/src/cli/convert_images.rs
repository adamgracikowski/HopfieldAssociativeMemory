//! The `convert-images` subcommand.

use std::path::PathBuf;

use clap::Args;
use compute::images::ConversionConfig;
use compute::patterns::summary::Summary;

#[derive(Debug, Args)]
pub struct ConvertImagesArgs {
    /// Conversion configuration (TOML) with the bitmap size, the source list,
    /// the image directory and the output datasets.
    #[arg(long)]
    config: PathBuf,

    /// Also write every bitmap as a black-and-white PNG to this directory.
    #[arg(long, value_name = "DIR")]
    preview: Option<PathBuf>,
}

impl ConvertImagesArgs {
    /// Converts the configured images, writes the datasets (and previews)
    /// and prints the statistics of each dataset.
    pub fn run(self) -> compute::Result<()> {
        let config = ConversionConfig::from_file(&self.config)?;
        for (i, conversion) in config.convert()?.iter().enumerate() {
            conversion.write()?;
            if let Some(dir) = &self.preview {
                conversion.write_preview(dir)?;
            }
            let file = &conversion.output.file;
            let name = file.file_stem().unwrap_or_default().to_string_lossy();
            let description = format!(
                "{} images, {} binarization",
                conversion.names.len(),
                conversion.output.method
            );
            if i > 0 {
                println!();
            }
            print!("{}", Summary::new(&name, &description, &conversion.set));
            println!("written to   {}", file.display());
        }
        Ok(())
    }
}
