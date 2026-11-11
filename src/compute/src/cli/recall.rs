//! The `recall` subcommand.

use std::path::PathBuf;

use clap::Args;
use compute::dynamics::Mode;
use compute::patterns::{Bitmap, PatternView};
use compute::run::{Input, OutputDir, RecallRun, RunConfig};

#[derive(Debug, Args)]
pub struct RecallArgs {
    /// Run configuration (TOML) with the dataset, the learning rule, the
    /// dynamics, the seed and the initial state.
    #[arg(long)]
    config: PathBuf,

    /// Output directory; it must be empty or not exist yet.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,

    /// Write into a non-empty output directory, replacing files of the same
    /// name.
    #[arg(long)]
    force: bool,

    /// Also print the initial and the final state as bitmaps.
    #[arg(long)]
    render: bool,
}

impl RecallArgs {
    /// Runs the configured recall, writes its artifacts and prints a
    /// summary.
    pub fn run(self) -> compute::Result<()> {
        let config = RunConfig::from_file(&self.config)?;
        let out = OutputDir::new(&self.out, self.force)?;
        let run = RecallRun::execute(&config)?;
        run.write(&config, &out)?;

        let dataset = &run.dataset;
        let summary = run.summary();
        let dynamics = &config.dynamics;
        println!(
            "dataset      {} ({}x{}, p = {}, N = {})",
            dataset.label(),
            dataset.width,
            dataset.height,
            dataset.patterns,
            dataset.neurons
        );
        println!(
            "learning     {}, {} backend, {} epochs",
            config.learning, run.training.backend, run.training.epochs
        );
        let order = match dynamics.mode {
            Mode::Asynchronous => format!(", {} order", dynamics.order),
            Mode::Synchronous => String::new(),
        };
        println!(
            "dynamics     {}{order}, sgn(0) {}",
            dynamics.mode, dynamics.zero_field
        );
        println!("input        {}", describe(&config.require_input()?));
        println!(
            "result       {} after {} steps, {} changes, energy {:.6}",
            summary.termination, summary.steps, summary.changes, summary.energy
        );
        if let (Some(target), Some(before), Some(after)) = (
            summary.target,
            summary.initial_distance,
            summary.final_distance,
        ) {
            println!("target       pattern {target}, distance {before} -> {after}");
        }
        println!(
            "nearest      pattern {}, overlap {:.4}",
            summary.nearest.pattern, summary.nearest.overlap
        );
        println!("frames       {}", summary.frames);
        println!("written to   {}", out.path().display());

        if self.render {
            let render = |state: PatternView<'_>| {
                Bitmap::new(state, dataset.width, dataset.height).map(|bitmap| bitmap.to_string())
            };
            print!("\ninitial state\n{}", render(run.initial.state.view())?);
            print!("\nfinal state\n{}", render(run.outcome.state.view())?);
        }
        Ok(())
    }
}

/// One-line description of the initial state.
fn describe(input: &Input) -> String {
    match input {
        Input::Pattern { index, flips } => {
            format!("pattern {index} with {flips} flipped pixels")
        }
        Input::Occlusion { index, rows } => {
            format!("pattern {index} with {rows} covered rows")
        }
        Input::Random {} => "random state".to_owned(),
        Input::File { path, index } => format!("pattern {index} of `{}`", path.display()),
    }
}
