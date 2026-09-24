//! Bodi: a System-1 engine for AI agents.
//!
//! * `memory` / `density` / `query`: fastmemory topology memory with a concept-density
//!   guard and a query cascade that reports how (and whether) each query matched.
//! * `system1`: Laya decisions executed natively via ONNX Runtime (`laya` feature),
//!   with shortlisting, order-ensembling, script gating and caching.
pub mod density;
pub mod engine;
pub mod error;
pub mod graph;
pub mod index;
pub mod ingest;
pub mod louvain;
pub mod memory;
pub mod query;
pub mod system1;
pub mod text;

pub use error::{BodiError, Result};
pub use engine::{Bodi, BodiConfig};
