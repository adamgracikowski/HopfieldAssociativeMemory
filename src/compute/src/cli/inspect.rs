//! The `inspect` subcommand.

use std::path::PathBuf;

use clap::Args;
use compute::Error;
use compute::patterns::registry::Registry;
use compute::patterns::summary::Summary;

#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Dataset name from the registry.
    name: String,

    /// Dataset registry (default: found from the current directory upwards).
    #[arg(long)]
    registry: Option<PathBuf>,

    /// Show only these patterns (indices from 0).
    #[arg(long = "pattern", value_name = "MU")]
    patterns: Vec<usize>,

    /// Print only the statistics, without bitmaps.
    #[arg(long, conflicts_with = "patterns")]
    no_render: bool,
}

impl InspectArgs {
    /// Prints the statistics and selected bitmaps of a dataset.
    pub fn run(self) -> compute::Result<()> {
        let registry = self.open_registry()?;
        let entry = registry.entry(&self.name)?;
        let set = registry.load(&self.name)?;

        let patterns = if self.no_render {
            Vec::new()
        } else if self.patterns.is_empty() {
            (0..set.len()).collect()
        } else {
            self.patterns
        };
        if let Some(&mu) = patterns.iter().find(|&&mu| mu >= set.len()) {
            return Err(Error::InvalidArgument {
                message: format!(
                    "pattern {mu} does not exist, `{}` has {} patterns",
                    self.name,
                    set.len()
                ),
            });
        }

        print!("{}", Summary::new(&self.name, &entry.description, &set));
        for mu in patterns {
            print!("\npattern {mu}\n{}", set.bitmap(mu));
        }
        Ok(())
    }

    /// Reads the given registry, or discovers it from the current directory.
    fn open_registry(&self) -> compute::Result<Registry> {
        match &self.registry {
            Some(path) => Registry::from_file(path),
            None => {
                let cwd = std::env::current_dir().map_err(|source| Error::Io {
                    path: PathBuf::from("."),
                    source,
                })?;
                Registry::discover(&cwd)
            }
        }
    }
}
