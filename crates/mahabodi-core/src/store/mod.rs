//! PostgreSQL as MahaBodi's built-in large-memory store (deploy/postgres/DESIGN_STORE.md).
//!
//! `derive` is pure Rust and always compiled: it turns ingested ATFs into the node, link and term rows the store
//! keeps, with the same rules as the in-process graph and index. The database layer is behind the `postgres`
//! cargo feature.

pub mod derive;
