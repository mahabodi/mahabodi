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
