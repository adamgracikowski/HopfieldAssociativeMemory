//! The `generate` subcommand.

use std::path::PathBuf;

use clap::Args;
use compute::generate::GroupsConfig;
use compute::patterns::summary::Summary;

#[derive(Debug, Args)]
pub struct GenerateArgs {
    /// Generation configuration (TOML) with the seed, the prototypes, the
    /// number of variants and the output datasets.
    #[arg(long)]
    config: PathBuf,
}

impl GenerateArgs {
    /// Generates the configured datasets, writes them with their provenance
    /// and prints the statistics of each dataset.
    pub fn run(self) -> compute::Result<()> {
        let config = GroupsConfig::from_file(&self.config)?;
        for (i, generated) in config.generate()?.iter().enumerate() {
            generated.write()?;
            let description = format!(
                "{} prototypes of {} x {} variants, {} flipped pixels each",
                config.prototypes.len(),
                config.dataset,
                config.variants,
                generated.flips
            );
            if i > 0 {
                println!();
            }
            print!(
                "{}",
                Summary::new(&generated.output.stem(), &description, &generated.set)
            );
            println!("written to   {}", generated.output.file.display());
        }
        Ok(())
    }
}
