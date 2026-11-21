//! Large strictly stable pattern sets: `S_u` holds every bipolar `xi` with
//! `u . xi = 0`. When `S_u` spans `u^perp`, Oja's subspace rule on a basis of
//! `S_u` gives every member the aligned field `u_i^2 / |u|^2 > 0`. The search
//! counts `|S_u|` for many `u` and checks the largest sets with both rules.

pub mod check;
pub mod config;
pub mod hebb_sum;
pub mod hyperplane;
pub mod metadata;
pub mod provenance;
pub mod run;

pub use check::StrictStability;
pub use config::StableConfig;
pub use hebb_sum::HebbSum;
pub use hyperplane::Hyperplane;
pub use metadata::StableMetadata;
pub use provenance::StableProvenance;
pub use run::{Candidate, StableSearch, Verified};
