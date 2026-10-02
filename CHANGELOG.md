# Changelog

## Unreleased (planned 0.2.0)

### Changed (behaviour)

- **`format="auto"` no longer parses fastmemory entity tags.** Prose with parenthesised forms such as `(Block 4)` stays
  a passage scoped to its `source`, and is no longer merged across documents. Entity tags are parsed only with
  `format="entity_tags"`, where identical tag names in different documents still merge into one ATF, by design.
  Anyone relying on auto-detected tags must now pass `format="entity_tags"`.

### Added

- **PostgreSQL store** (cargo feature `postgres`). The `store_*` methods go through `call()`; the feature is
  compiled into the Python binding, and the Node, Java, C# and Go crates gain the same opt-in feature in 0.2.0
  (until then their builds did not include the store at all; store tests are written for all four, not yet run).
  Tested from Rust and Python so far. Memory
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
  - **No-match behaviour differs from in-process, by design:** the store has no hub fallback. A query nothing
    matches returns `stage: empty_memory`, no hits and `handoff: true`, where in-process memory would return hub
    neighbours. Agents should branch on `handoff`, which is consistent across both.
  - **TLS:** via rustls (no OpenSSL). By default the server certificate is verified against the Mozilla root store,
    so managed providers whose certificates chain to a private CA (AWS RDS/Aurora, Google Cloud SQL, some Azure
    setups) fail verification; providers with public-CA certificates work. New: `sslrootcert=<pem>` in the DSN
    verifies against that CA bundle instead, like libpq (`sslmode=require` + `sslrootcert` behaves like libpq's
    `verify-full`: rustls always checks the certificate and hostname when TLS runs). The default `sslmode=prefer`
    proceeds **unencrypted** when the server does not offer TLS — use `require` to guarantee encryption. A DSN
    without `sslrootcert` passes through byte-identical (quoted values, spaces and escapes untouched); with it,
    only that key=value pair is removed, per libpq's conninfo grammar.
  - **Fork safety:** a forked child opens its own connections. This is tested on macOS arm64; Linux is not yet tested.
  - **Binding store tests:** every binding suite (Python, Node, Java, C#, Go) carries a store round-trip test
    (gated on `MAHABODI_TEST_PG_DSN`) plus a non-skipping check that the `postgres` feature is really compiled in
    (an unreachable-DSN `store_open` must fail with a store error, never "unknown method"). `scripts/test_all.sh`
    and the release builds compile every binding with the feature.

### Fixed

- **In-process BM25 used an imprecise average passage length for very large memories.** The corpus-length average
  was summed in f32.
  - Node lengths are whole numbers, so that sum is exact until the total passes 2^24 (16,777,216), about 750K nodes
    of typical passages.
  - Past that point it drifts, by an order-dependent amount: +0.5 % when a 100K-page entity-linking pool is summed
    in one tested order, and up to +2.7 % in others. That skews every BM25 length norm slightly.
  - It is now accumulated in f64. In-process lexical scores change only for memories past that size.
  - Found by the PostgreSQL store's parity diagnostics: the store always computed this value exactly.

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
