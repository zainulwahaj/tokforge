//! Shared tokenization and source-preserving document preparation.

pub mod api;
pub mod batch;
pub mod bytelevel;
mod engine;
pub mod work;

pub use engine::{
    Artifact, Chunk, CoreMode, EncodeOptions, Encoded, Encoding, Engine, Error, Span,
    SpecialTokens, BACKEND_CORE,
};

/// Version reported by all interfaces.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
