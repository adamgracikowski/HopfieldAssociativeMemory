//! Single runs configured by a TOML file: training a network and recalling
//! a pattern from an initial state.

pub mod config;
pub mod input;
pub mod learning;

pub use config::RunConfig;
pub use input::{Initial, Input};
pub use learning::Learning;
