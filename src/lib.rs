//! Shared tokenization and source-preserving document preparation.

pub mod api;
pub mod batch;
mod engine;
pub mod work;

pub use engine::{Artifact, Chunk, EncodeOptions, Encoded, Engine, Error, SpecialTokens};

/// Version reported by all interfaces.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
