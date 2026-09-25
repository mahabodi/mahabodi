//! MahaBodi: a System-1 engine for AI agents.
//!
//! This crate re-exports [`mahabodi_core`], so `cargo add mahabodi` gives the whole engine:
//! fastmemory-style topology memory (ingest, density guard, query cascade, hybrid retrieval) and
//! typed decisions from Laya models through ONNX Runtime (feature `laya`, on by default).
//!
//! ```no_run
//! use mahabodi::engine::Bodi;
//! let mut b = Bodi::new(Default::default()).unwrap();
//! ```
//!
//! Results and caveats: <https://github.com/mahabodi/mahabodi/blob/main/BENCHMARKS.md>.
pub use mahabodi_core::*;
