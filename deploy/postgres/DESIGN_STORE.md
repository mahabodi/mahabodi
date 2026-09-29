# Design: PostgreSQL as MahaBodi's built-in large-memory store (planned for 0.2.0)

Status: design, revised after review, not implemented. It replaces the research scripts `research/el_pg_load.py` and
`research/bench_el_pg.py` with an engine feature.

## What will be claimed

The claim will read: "A built-in PostgreSQL backend for **memory retrieval and grounded decisions** at millions of
passages. You bring a PostgreSQL 16/17 server with pgvector and pg_trgm; a one-line Docker image is provided."

It is never described as zero setup, and never claimed to cover features the store doesn't implement (see
Coverage).

## API (engine `call()` methods; every binding gets them)

| Method | Arguments | Does |
|---|---|---|
| `store_open` | `dsn`, `namespace`, `create` (default true) | Connect (TLS via rustls). Apply the schema if missing (embedded, versioned migrations). Create the namespace. The DSN is never logged. |
| `store_ingest_batch` | `docs: [{text, source, format}]`, `batch` | Split with the existing `ingest::ingest` (identical passages to in-process). Compute `text::terms()` and `text::stem()` per passage in Rust. Embed each passage with the loaded embedder. COPY in one transaction per batch; resumable, since a batch whose first id exists is skipped. Update the term and fuzzy vocabularies. |
| `store_build_index` | `workers`, `maintenance_mem` | GIN on `terms` and `stems`, a trigram GIN on `id`/`label`, HNSW on vectors. |
| `store_query` | `q`, `k` | The cascade below. It returns the same shape as in-process `query` (hits, stage, matched, handoff, confidence, coverage). |
| `store_context` / `store_decide_with_memory` | as in process | Store retrieval, then the unchanged `decide`. |
| `store_stats` | none | Rows, vectors, index state, schema version. |

## Store-side cascade: the same rules, computed by the same code

MahaBodi's `Index` rules are applied in Rust, on both sides, not approximated with PostgreSQL text configs. The
review found those differ in tokenisation, stopwords, stemming and match semantics.

Columns stored per passage:

- `terms text[]`: `text::terms()` of the passage, computed in Rust. It uses MahaBodi's own tokeniser, which splits
  camelCase and letter/digit boundaries and bigrams unspaced scripts, and its own stopword list.
- `stems text[]`: `text::stem()` over those terms, the same light stemmer with the same exception list.
- `id`, `label`: for fastmemory's substring rule.
- `embedding`: per passage.

| In-process stage (query.rs) | Store implementation |
|---|---|
| exact: any query term matches; score and coverage as in `search_exact` | query terms from `text::terms()` in Rust. Candidates: `terms && $q` (GIN). Score: a port of `search_exact`'s weighting (term rarity from the stored document frequencies), with coverage reported. |
| substring: the whole query in node id/label (fastmemory's rule, lowercased) | `lower(id) LIKE '%q%' OR lower(label) LIKE '%q%'` on a trigram GIN over id/label, not the body |
| stem | `stems && $stems`, scored as in process |
| fuzzy | the trigram vocabulary built from the same stored terms; corrected terms are re-run through exact and stem |
| dense | pgvector HNSW over per-passage vectors, fused by reciprocal rank as in `query_with` |
| hub | **none** in the store backend. Nothing matching is a handoff (see Differences). |

Unit tests on a small corpus require the in-process `Index` and the store to return the same stage and the same
top-k for a fixed query set. Any difference fails the test.

**Graph nodes are part of ranking, not just passages.** The in-process `Index` scores every graph node: passage ATFs,
and the Data / Concept / Access / Event nodes built from `data_connections` and density concepts, whose labels
carry field weight 3.0. A matched non-passage node spreads `0.5 · s / sqrt(deg)` to the passages linked to it
(query.rs). BM25's N and average length count all nodes. So the store also keeps:

- a `node` table per namespace: name, level, label terms and stems (from `text::terms` and `text::stem`), degree;
- the passage ↔ node links, taken from `data_connections` and density concepts;
- the same spreading step, done in SQL over those links;
- BM25 statistics (N, average length, document frequencies) over all nodes, exactly as `Index::build` computes them.

The Louvain *blocks* (communities) still don't rank; they only add sibling context. The AGE projection isn't needed
for ranking: plain relational tables are enough.

## Differences from in-process (documented, not hidden)

- **Density runs when asked.** The engine runs density after every `ingest_batch` call. The store runs it only on
  `store_ensure_density`, so bulk loads can write many batches and densify once. To match an engine that received
  the same documents in N calls, call `store_ensure_density` after each of those N loads. The benchmark and the dev
  parity gate load the pool in one logical ingest (as `bench_el.py` did, one `ingest_batch`) and then densify once.

- **No hub fallback.** In-process `query` never returns empty; the store returns an empty handoff instead.
- **Ranking at scale** may still differ in tie order and in the approximation of HNSW vs exact search. The 100K
  bridge measures it, with a pre-registered parity gate (see Evaluation).

## Coverage (what the store backend does and doesn't do)

- **Covered in 0.2.0:** ingest, query, context, `decide_with_memory`.
- **Not covered in 0.2.0:** experience memory (`learn()`, `calibrate`) stays in process. The store's docs say this.
  A store-backed experience memory is a later item.
- **Not a ranking signal:** the AGE graph. It is optional and used for traversal only. In-process ranking doesn't use
  the graph either.

## Client, build and safety

- **Client:**
  - the sync `postgres` crate with **rustls** TLS (no OpenSSL, safe for the glibc 2.28 floor), enabled in shipped
    builds so managed PostgreSQL works (RDS, Cloud SQL, Supabase, Neon);
  - a small pool (`r2d2_postgres`), because the client isn't `Sync` and `Bodi` is shared across threads in the
    bindings.
- **Fork safety:**
  - the client embeds a tokio runtime, so it is **not fork-safe**;
  - the docs say to open the store after `fork()` (Python multiprocessing);
  - a stored process id makes `store_*` reconnect in a forked child.
- **Packaging:**
  - a cargo feature `postgres` in `mahabodi-core`, which the default in-process build doesn't include;
  - wheel, npm and NuGet builds turn it on after it passes on Linux and macOS;
  - the glibc 2.28 `objdump` check is kept for every binary;
  - wheel size is reported.
- **Where it's out of the box:** only where prebuilt packages ship it (PyPI, npm, NuGet, crates.io). Java and Go stay
  source builds.
- **Server:** a published Docker image (the tested `deploy/postgres/Dockerfile`, with a `docker run` line in the docs),
  plus the native setup that was used on the Mac mini.

## Tests

- **Unit tests (no server):** term and stem extraction parity, cascade fusion, SQL generation, and Index vs store
  parity on an in-memory fixture.
- **Integration tests,** gated on `MAHABODI_TEST_PG_DSN`:
  - an ingest → query round trip;
  - passage parity with `snapshot()`;
  - resume after an interrupted batch;
  - the tag-lookalike regression;
  - the TLS connect path;
  - reconnect after `fork`.
- **Binding suite:** `store_open` / `store_query` in all bindings, against the Mac mini's PostgreSQL.

## Evaluation (pre-registered separately, before running)

1. **Parity gate first, on the DEV 100K pool with the 500 dev mentions**, never on test data.
   - Compared against the in-process M dev numbers (`bench_el_tune.json`, k = 20, n = 48).
   - Pass: store shortlist recall within 3 points of in-process, **and** accuracy a McNemar tie.
   - Iterating on the implementation until it passes happens on dev only.
   - The test 100K bridge then runs **once**, after the gate passes, and is reported as is.
   - If the gate can't be passed, the 5.9M v2 run is not presented as "MahaBodi at 5.9M" until that is explained in a
     dated clarification.
2. **5.9M v2 with the shipped feature:** a 2 × 2 ablation (per-passage vectors on/off × MahaBodi cascade on/off),
   with p50 and p95 latency next to every accuracy.
3. **Ingest cost:** throughput is measured first. Per-passage embedding of ~23M passages is reported with hardware and
   wall time. An earlier note called it infeasible on the Ubuntu CPU; it gets measured on the Mac mini (CPU and MPS)
   before it is promised.
