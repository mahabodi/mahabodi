# Changelog

## Unreleased (planned 0.2.0)

### Changed (behaviour)

- **`format="auto"` no longer parses fastmemory entity tags.** Prose with parenthesised forms such as `(Block 4)` stays
  a passage scoped to its `source`, and is no longer merged across documents. Entity tags are parsed only with
  `format="entity_tags"`, where identical tag names in different documents still merge into one ATF, by design.
  Anyone relying on auto-detected tags must now pass `format="entity_tags"`.

### Added

- **PostgreSQL store** (cargo feature `postgres`). It is available through every binding's `call()` (`store_*`
  methods) and as Python methods, and is tested from Rust and Python only so far. Memory
  too large for one process lives in PostgreSQL 16/17, with pgvector and pg_trgm. It stores the same passages, graph
  nodes and term statistics as in-process memory and searches with the same cascade. See
  [deploy/postgres/DESIGN_STORE.md](deploy/postgres/DESIGN_STORE.md).
  - **Retrieval parity:**
    - on test fixtures, the same top-20 results, scores, stage and handoff as the in-process engine;
    - at 100K pages, the pre-registered dev check matched in-process accuracy and shortlist recall. That check ran at
      cc3a70a, with exact vector search and before store schema 2;
    - the fixture parity tests pass on the current code.
  - **Vectors:** per-passage, as `vector` (float4) or `halfvec` (fp16). Index with HNSW (`store_build_vector_index`,
    `store_set_ef_search`) or IVFFlat (`store_build_ivfflat_index`, `store_set_probes`).
  - **Query modes:** `hybrid` (default), `lexical` and `dense`.
  - **TLS:** via rustls (no OpenSSL). The refusal path is tested; a handshake with a certificate-verified server is not
    yet tested.
  - **Fork safety:** a forked child opens its own connections. This is tested on macOS arm64; Linux is not yet tested.

### Fixed before release (found by tests and review; never shipped)

- **Namespace isolation of vector indexes.** Namespaces shared one vector table and one index name, and the index
  build used `IF NOT EXISTS`. So a second namespace silently reused the first one's index. Queries also filtered
  across namespaces after the index scan, and the first writer fixed the vector type for all namespaces.
  - Each namespace now has its own vector table and index, with its own vector type (store schema 2).
  - A rebuild replaces the namespace's existing index and reports the old definition.
  - A store opened with another vector type is refused.
- **Dense queries always use a built vector index.** PostgreSQL could switch to an exact sequential scan when the
  search setting made the index look costlier. Dense queries now disable sequential scans for their own transaction.
- **`store_query(mode="dense")` now uses the vector index.** It ordered the vector–node join by distance, so PostgreSQL
  joined every vector and sorted. It now runs the nearest-neighbour search first, then joins only those hits.
