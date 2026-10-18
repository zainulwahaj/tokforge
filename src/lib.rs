//! Shared tokenization and source-preserving document preparation.

mod engine;

pub use engine::{Artifact, Chunk, EncodeOptions, Encoded, Engine, Error, SpecialTokens};

/// Version reported by all interfaces.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
