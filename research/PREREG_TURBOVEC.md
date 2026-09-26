# Pre-registration: memory compression vs retrieval (FastMemory, MahaBodi, TurboVec)

Written and agreed with the reviewer **before any run**. It tests the claim in fastmemory's README table
("Production Memory Efficiency (Verified)", commit a7dec44): "35x compression … no information lost" against
TurboVec's "~4 GB / 8x". That table describes different data in its two columns and a computed, not measured,
TurboVec size. Here every arm gets the **same corpus** and is measured on the **same machine**.

## Question

For the same documents, how many bytes does each memory store, and what does it give back:
- can it find the right document for a held-out query;
- can the original text and ids be recovered from what it stores?

A compression ratio is **never reported alone**, only next to recall and recoverability.

## Corpus and sizes

- **KILT Wikipedia pages** (`/media/sda/data/kilt/pages`, built by `kilt_prep.py`): the first 10K, 30K and 100K
  pages of `part-0000`, the same subsets as `probe_memory_scale.py`.
- **Indexed text per page:** `# <title>\n\n<abstract>` (abstract ≤ 2,000 characters), exactly as in the ingest
  probe.
- **Raw source bytes:** UTF-8 bytes of title + abstract.
- **Optional second corpus: PubChem**, 100K compounds, text fields as FastMemory ingests them. Only if a
  public, reproducible download is found; otherwise reported as "not run", with the reason.

## Queries (held-out, known gold, no labels needed)

Up to 1,000 pages per size (seed 0) that have a **paragraph after the abstract** (not indexed by any arm). Per
page, three queries whose gold answer is that page:
1. **Title query:** the page title. This is an easy control, since the title is indexed.
2. **Held-out sentence:** the first sentence of the first non-abstract paragraph, 8–40 words.
3. **Keyword query:** the 3 rarest content terms of that held-out paragraph, using document frequency over the
   indexed corpus. Terms that do not occur in the indexed corpus are skipped; if fewer than 2 remain, there is no
   keyword query for that page.

The number of queries of each type is reported.

## Arms

| Arm | What is stored | How it searches |
|---|---|---|
| F: FastMemory topology | The JSON from `fastmemory.process_markdown` (source-built at a7dec441, as in `probe_fastmemory.py`), with pages converted to ATF exactly as in that script | fastmemory's own search rule, as ported in `mahabodi_core::query::fastmemory_search` (substring match on block/node name, action, id). A hit counts as page P only if the returned block contains a node whose id or action identifies P (`ATF_<id>` or `Process_<title>`), since the topology has no other link back to pages |
| M: MahaBodi memory | MahaBodi's in-process memory, via `ingest_batch` (text, graph, index) | `query(q, k)`, lexical cascade. Also a second row with the MiniLM embedder loaded (hybrid), if memory allows |
| V32: MiniLM float32, flat | 384-d MiniLM vectors (the model MahaBodi uses), float32, exact inner product | dense top-k |
| TV4 / TV2: TurboVec 4-bit / 2-bit | The same MiniLM vectors in `turbovec.TurboQuantIndex(dim=384, bit_width=4 or 2)` | `index.search(q, k)` |

The vector arms embed the same `# <title>\n\n<abstract>` text; the queries are embedded with the same model.

## Measurements per arm and size

- **Stored bytes on disk:**
  - F: the topology JSON;
  - M: not serialisable today, so peak RSS stands in, and this is disclosed;
  - V32: the float32 matrix;
  - TV: the `.tv` file from `index.write`.
- **Vector indexes need the page text alongside** to return it, so their bytes are reported both **without text**
  and **with the texts and ids** (the raw text + id bytes).
- **Ratio:** raw source bytes ÷ stored bytes, for each variant.
- **Peak RSS** while serving queries (a separate process per arm and size).
- **Build time:** embedding time for the vector arms, reported separately from index build time.
- **Retrieval:** recall@1, @5 and @10 per query type, with Wilson 95 % CIs; p50 and p95 query latency, CPU only,
  8 threads.
- **Recoverability:** the fraction of the 1,000 sampled pages whose full indexed text, and separately whose id,
  can be recovered from what the arm stores.
  - F: search the JSON verbatim, as in `probe_fastmemory.py`.
  - M: `text_of`.
  - Vector arms without text: 0 by construction; with text: 1.

## Rules

- Runs on the Ubuntu CPU (`CUDA_VISIBLE_DEVICES=""`); the GPU stays with the KILT index build. Embeddings for the
  vector arms are computed on CPU as well, or reused from the shared KILT dense index for exactly the same
  pages and text, if that index is finished (the text must match; checked by a per-page hash).
- Every arm is reported, including arms that fail or run out of time (reason recorded).
- Per-item results (ranks per query per arm) are saved so the reviewer can recompute everything.
- **Wording is bound to the numbers:** no "Nx compression" unless measured on this corpus, and always stated
  together with that arm's recall and recoverability. The reviewer recomputes the results from the committed files
  before anything is published.

## Pre-run clarifications (2026-09-26, before any query was scored; agreed with the reviewer)

**1. Which fastmemory is tested (arm F).** A check against fastmemory's own documented inputs, before scoring:
- The Python module (`fastmemory.process_markdown`, source-built at a7dec441) ignores ATF structure, even for the
  ATF example in fastmemory's own README: `## [ID: auth_module]` becomes nodes `F_*` / `D_Auth_module` with action
  "Extrapolated". On `example/world_events/input.md` (20 ATFs) it keeps 0/20 ids.
- fastmemory's shipped `example/world_events/output.json` keeps 11/20 ids. It was made by the CLI (`run.sh`:
  `cargo run -- input.md`).
- The CLI does not compile at a7dec441 or 05d1e63 (`run_louvain` not found). It does compile at **64cb29b**
  (2026-03-29), which execs an embedded `rust-louvain` helper.
  - That helper is 0 bytes for Linux and Windows (also in the PyPI 0.4.6 sdist, sha256 cd2478d6…), so nothing runs
    on Linux.
  - The macOS helper is a universal binary (x86_64 + arm64). On macOS x86_64 the 64cb29b CLI reproduces the shipped
    world_events output: same 10 Function nodes, 11/20 ids.
- So:
  - **Primary F** = the fastmemory CLI at 64cb29b on macOS x86_64
    (`research/bench_turbovec_fastmemory_cli.py`).
  - **Secondary Fpy** = the Python module on Ubuntu, labelled "different code path". Its recall is reported as
    0.00 **together with** the structural reason (no page identity in its output).
- F's build time, latency and RSS are from a different machine: reported, flagged "not comparable". Stored bytes,
  recoverability and recall do not depend on the machine.

**2. No telemetry.** The CLI pings a license-telemetry server on every run. Every CLI run is under macOS
`sandbox-exec` with `(deny network*)`, so the ping cannot leave the machine. With network denied, the CLI builds the
same structure (same Function set, same id count; the byte order differs because fastmemory's Louvain breaks ties
by HashMap order).

**3. F search.**
- F searches with fastmemory's `query::search_memory` (`src/query.rs` lines 3–85 at 64cb29b), ported line for line
  (Python in the F runner, Rust in `mahabodi_core::query::fastmemory_search`).
- Parity is checked, not assumed: for 25 queries per size, the port's Function-id set is compared with the real
  sandboxed `fastmemory query` CLI's. The CLI re-clusters on every call, so CLI-vs-CLI agreement between two runs is
  reported too.
- If the port disagrees beyond the CLI's own run-to-run variation, the CLI's results are used.
- Scoring: pages are ranked by their first appearance (`F_ATF_<wikipedia_id>`) in the result (it has no scores);
  recall@k counts the gold page among the first k.

**4. MahaBodi (arm M) stored bytes** = `snapshot()` JSON bytes (atfs, links, texts, concepts; the graph and index are
rebuilt on restore), also reported **without the `texts` field**, for comparison with the "without text" numbers of
the vector arms. Recoverability is checked from the snapshot's texts: every sentence of the abstract must be found in
the page's texts. RSS is still reported.
