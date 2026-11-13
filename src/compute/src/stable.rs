//! Large sets of patterns that are all strictly stable in the trained
//! network, built from hyperplanes of the state space.
//!
//! For an integer vector `u` without zeros, `S_u` holds every bipolar state
//! `xi` with `u . xi = 0`. Oja's subspace rule trained on a basis of `S_u`
//! tends to the projection `P` onto the span of `S_u`; when the members
//! span the whole hyperplane `u^perp`, the zeroed diagonal gives every
//! member the aligned field `xi_i h_i = 1 - P_ii = u_i^2 / |u|^2 > 0`. The
//! search counts `|S_u|` for many `u` and checks the strict stability of
//! the largest sets with both rules, without storing their members.

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
