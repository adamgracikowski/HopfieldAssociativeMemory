//! Single runs configured by a TOML file: training a network and recalling
//! a pattern from an initial state.

pub mod config;
pub mod dataset;
pub mod input;
pub mod learning;
pub mod metadata;
pub mod output_dir;
pub mod recall;
pub mod train;

pub use config::RunConfig;
pub use dataset::DatasetInfo;
pub use input::{Initial, Input};
pub use learning::Learning;
pub use metadata::{Metadata, TrainingInfo};
pub use output_dir::OutputDir;
pub use recall::{Frames, Nearest, RecallRun, RecallSummary};
pub use train::TrainRun;
