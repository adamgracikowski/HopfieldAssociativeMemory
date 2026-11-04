//! Compute layer of the Hopfield associative memory project.
//!
//! The crate trains Hopfield networks with the Hebbian and Oja's learning
//! rules, simulates their synchronous and asynchronous dynamics and writes
//! the results to files that the presentation layer visualizes.

pub mod build_info;
pub mod dynamics;
pub mod error;
pub mod images;
pub mod learning;
pub mod patterns;
pub mod weights;

pub use error::{Error, Result};
pub use patterns::PatternSet;
