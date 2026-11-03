//! Command-line interface of the compute layer.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use compute::images::ConversionConfig;
use compute::patterns::registry::Registry;
use compute::patterns::summary::Summary;
use compute::{Error, build_info};

/// Hopfield associative memory: training, recall and experiments.
#[derive(Debug, Parser)]
#[command(name = "compute", version = build_info::LONG_VERSION, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Show the size, statistics and bitmaps of a registered dataset.
    Inspect(InspectArgs),

    /// Convert photographs to bipolar bitmap datasets saved as `.npy`.
    ConvertImages(ConvertImagesArgs),
}

#[derive(Debug, Args)]
struct InspectArgs {
    /// Dataset name from the registry.
    name: String,

    /// Dataset registry. By default `data/datasets/datasets.toml` is searched
    /// for in the current directory and its parents.
    #[arg(long)]
    registry: Option<PathBuf>,

    /// Show only these patterns (indices from 0). By default all are shown.
    #[arg(long = "pattern", value_name = "MU")]
    patterns: Vec<usize>,

    /// Print only the statistics, without bitmaps.
    #[arg(long, conflicts_with = "patterns")]
    no_render: bool,
}

#[derive(Debug, Args)]
struct ConvertImagesArgs {
    /// Conversion configuration (TOML) with the bitmap size, the source list,
    /// the image directory and the output datasets.
    #[arg(long)]
    config: PathBuf,

    /// Also write every bitmap as a black-and-white PNG to this directory.
    #[arg(long, value_name = "DIR")]
    preview: Option<PathBuf>,
}

fn main() -> miette::Result<()> {
    match Cli::parse().command {
        Command::Inspect(args) => inspect(args)?,
        Command::ConvertImages(args) => convert_images(args)?,
    }
    Ok(())
}

/// Prints the statistics and selected bitmaps of a dataset.
fn inspect(args: InspectArgs) -> compute::Result<()> {
    let registry = open_registry(args.registry)?;
    let entry = registry.entry(&args.name)?;
    let set = registry.load(&args.name)?;

    let patterns = if args.no_render {
        Vec::new()
    } else if args.patterns.is_empty() {
        (0..set.len()).collect()
    } else {
        args.patterns
    };
    if let Some(&mu) = patterns.iter().find(|&&mu| mu >= set.len()) {
        return Err(Error::InvalidArgument {
            message: format!(
                "pattern {mu} does not exist, `{}` has {} patterns",
                args.name,
                set.len()
            ),
        });
    }

    print!("{}", Summary::new(&args.name, &entry.description, &set));
    for mu in patterns {
        print!("\npattern {mu}\n{}", set.bitmap(mu));
    }
    Ok(())
}

/// Converts the configured images, writes the datasets (and previews) and
/// prints the statistics of each dataset.
fn convert_images(args: ConvertImagesArgs) -> compute::Result<()> {
    let config = ConversionConfig::from_file(&args.config)?;
    for (i, conversion) in config.convert()?.iter().enumerate() {
        conversion.write()?;
        if let Some(dir) = &args.preview {
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

/// Reads the given registry, or discovers it from the current directory.
fn open_registry(path: Option<PathBuf>) -> compute::Result<Registry> {
    match path {
        Some(path) => Registry::from_file(&path),
        None => {
            let cwd = std::env::current_dir().map_err(|source| Error::Io {
                path: PathBuf::from("."),
                source,
            })?;
            Registry::discover(&cwd)
        }
    }
}
