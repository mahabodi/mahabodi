# Pre-registration: entity linking at 5.9M pages with the shipped PostgreSQL store (v2)

Written 2026-09-29 EDT. This is the store itself (`deploy/postgres/DESIGN_STORE.md`), not research scripts. Nothing
in here has been run on the store yet.

## Why a v2

The v1 5.9M run (PREREG_MILLION_SCALE.md, clarifications 4b/4c) used a simplified PostgreSQL retriever: one vector
per page, PostgreSQL's own text ranking and no graph. At 100K it cut shortlist recall from 0.677 to 0.479. v2 runs
MahaBodi's real retrieval through the store:

- the same passages;
- the same node graph and weighted terms;
- the same exact → substring → stem → fuzzy cascade with node spreading;
- density;
- per-passage vectors fused by reciprocal rank.

Parity with the in-process engine is shown by `crates/mahabodi-core/tests/store_pg.rs` on fixtures: top-20 ids and
scores, stage, coverage and handoff.

## Step 1: parity gate on DEV (iterating here is allowed; test data is never used)

- **Data:** the dev 100K pool (`pool_dev_100000_mq.npy`) and the 500 dev mentions from `mentions_mq.jsonl`, the same
  as bench_el tuning.
- **Reference:** in-process M at the selected setting (k = 20, n = 48), from `bench_el_tune.json` (per-item preds and
  shortlist hits; the Mac mini reproduced them 500/500).
- **Store run:**
  - `store_ingest_batch` of the pool pages (`# <title>\n\n<abstract>`, source `pg<row>`), in batches;
  - `store_build_index`;
  - ONE `store_ensure_density` pass, matching bench_el's single `ingest_batch`;
  - exact dense search (no HNSW at 100K, as in process);
  - retrieval by the mention string via `store_query(k = 60)`, mapped to pages and truncated to 20, exactly like
    `m_short`;
  - the unchanged `decide` with n = 48.
- **Pass:** store shortlist recall within 3 points of in-process (0.708 ± 0.03) **and** accuracy vs in-process M an
  exact-McNemar tie (p ≥ 0.05).
- **Also reported:** per-item agreement of shortlists (Jaccard of the top 20), load, density and query times, and the
  number of density concepts.
- **If it fails:** fix the store and re-run on dev only. Each attempt is recorded, and the final passing run is the
  gate.
- **If the gate can't be passed:** step 2 doesn't run as "MahaBodi at 5.9M"; a dated note explains why.

## Step 2: HNSW setting, chosen on DEV before any test run

- At 5.9M, exact dense search over ~23M passage vectors is too slow per query, so the v2 run uses the HNSW index.
- Its search setting is chosen on dev at 100K:
  - dense recall@50 of HNSW vs exact search for `ef_search` ∈ {40, 100, 200, 400}, at m = 16, ef_construction = 64;
  - the smallest `ef_search` with recall@50 ≥ 0.98 is selected (or 400 if none reaches it);
  - both recall and latency are reported.

## Step 3: test (once, after steps 1–2 pass)

- **Bridge on the test 100K pool:**
  - store vs in-process M (bench_el.json per-item preds): exact McNemar and shortlist recall;
  - with exact dense search, then with HNSW at the selected setting.
- **The 5.9M run:** the same 1,000 test mentions as bench_el. *[Superseded by clarification 1: the primary 5.9M result uses a fresh 1,000-mention sample; these items are the secondary "same items as v1" row.]*
- **2 × 2 descriptive ablation:** per-passage vectors on/off × lexical cascade on/off. "Cascade on" means the *[Made precise by clarification 2.]*
  lexical stages plus node spreading.
- **Primary:** store M at 5.9M vs Laya + dense shortlist (L1 from bench_el.json, the same items): accuracy with a
  Wilson CI and exact McNemar.
- **Reported next to every accuracy:**
  - shortlist recall;
  - latency p50/p95;
  - load, density and index-build times and hardware.
- **A spreading on/off factor** runs on dev only, descriptively (the reviewer's suggestion), never folded into the
  test.
- **Density cost:**
  - the dev 100K density time is measured in step 1 and extrapolated before step 3;
  - if density at 5.9M is capped or skipped, that is a dated clarification before step 3, and "cascade on" at 5.9M
    says so.

## Rules

- **Machine:** the same Mac mini (M2 Pro, 16 GB), one benchmark at a time.
- **Records:** provenance and a whole-tree source hash in every result file.
- **Review:** the reviewing agent recomputes every number from per-item files before anything is cited.
- **Losses** are reported as losses.

## Clarification 1 (2026-09-29 EDT, before any v2 step has run): a fresh test sample for the primary result

- **Why:** v2 exists because of what v1's test showed (low PostgreSQL recall). A v2 result on the same 1,000 test
  mentions would be a second attempt on the same items.
- **Fresh sample:**
  - drawn from `aidayago2-dev-kilt.jsonl`, the same file as the v1 test sample. It has 4,784 items.
  - Items whose gold page is missing, the 1,000 v1 test mentions and the 500 dev mentions (from `aidayago2-train`)
    are excluded, leaving 3,784 eligible.
  - `random.Random(2)` shuffles the eligible ids in file order and takes the first 1,000.
  - Overlap: 0 with v1 test, 0 with dev.
  - The ids are in `research/results/el_fresh_v2_ids.json`. SHA-256 of the sorted ids joined by newlines:
    `92f19ef9d38678ad218dfb4e5eacd88efc44cbbd0debeb3ffff2ea10437aa9df`.
  - Training mentions can't be used, because they feed the alias prior. KILT's AIDA test labels are hidden.
- **Primary v2 result:** store M vs L1 (Laya + dense shortlist, re-run on these items with bench_el's settings) at
  5.9M pages, on the **fresh** 1,000. Accuracy with a Wilson CI, exact McNemar, shortlist recall, latency.
- **Secondary:** the same comparison on the v1 test 1,000, labelled "same items as v1", for continuity only.
- **Unchanged:** the 100K bridge stays on the v1 test pool and mentions. Its purpose is store-vs-in-process parity on
  identical items, not a headline.
- **Fresh-item preparation:** state and mention extraction, and the dense top-k for L1, exactly as
  `kilt_el_prep.py` does (mention query).

## Record: step 1 result and step 3 costs (2026-09-29 EDT, before steps 2 and 3)

- **Step 1 passed on attempt 1** (`research/results/store_parity_gate_attempt1.json`, provenance `cc3a70a6`).
  - Store accuracy 0.198 [0.165, 0.235] and shortlist recall 0.708, equal to in-process M (0.198 / 0.708).
  - McNemar store vs in-process: 1/1, p = 1.0.
  - Shortlist hits agree on 500/500. Predictions agree on 488/500: 10 of the 12 differences are both wrong, and there
    is 1 flip each way. The non-gold shortlist order differs on those items; the cause is not verified.
- **Measured costs on the Mac mini at 100K pages (352,656 passages):**
  - load, including a MiniLM vector per passage on the CPU: 1,619 s;
  - index build: 97.5 s;
  - density: 464.9 s (3 rounds, 647,360 concept edges);
  - query + decide: p50 2.09 s, p95 2.57 s.
- **Extrapolation to 5.9M pages** (~23M passages, ~65×), if costs are linear:
  - load ~29 h;
  - density ~8.4 h.
- **Plan if density turns out super-linear:**
  - measure density at 5.9M with a hard wall-clock budget of 24 h;
  - if it can't finish, or needs more memory than the Mac mini has, cap it with a dated clarification written before
    any 5.9M query is scored (for example rounds = 1, or density only on under-linked passages);
  - "cascade on" at 5.9M then states the cap.
- The 5.9M store is loaded once and reused for every arm and ablation.

## Clarification 2 (2026-09-29 EDT, before any 5.9M query is scored): what the 2 × 2 ablation varies

The ablation separates the two v1 simplifications. "Vectors off" does **not** mean "no dense retrieval". Its four
cells:

| | MahaBodi cascade (store lexical stages + node spreading) | PostgreSQL text ranking (v1: `mahabodi_pg.search` lexical part, `ts_rank_cd` + pg_trgm correction) |
|---|---|---|
| **Per-passage vectors** (store `vec`) | the full v2 store (hybrid) | v1 lexical + per-passage dense |
| **First-passage-only vectors** (v1 layout: each page's existing KILT vector, `title + '. ' + abstract`) | store cascade + v1 dense | **v1 exactly** (the reproduced v1 M_pg retriever) |

- **Fusion:**
  - every cell fuses its lexical and dense lists by reciprocal rank (k = 60, lexical top 50 and dense top 50, ties
    by id), as `query_with` does;
  - the store's own path does this in Rust;
  - the mixed cells do it in the harness, with the same formula;
  - v1's corner uses `mahabodi_pg.search` unchanged.
- **Sources:** the v1 corner and the v1 lexical and dense lists come from the v1 5.9M database (`el_full`), which is
  unchanged since v1.
- **Downstream, identical in all cells:** the shortlist rule (k = 60 hits → pages → 20) and the decider (`decide`,
  n = 48).
- **Items:** the primary fresh sample. It is descriptive; the primary claim stays the full store vs L1.
- **Supplementary:** dense-only and lexical-only store runs (`store_query` modes).
- **Check for the v1 corner** (added 2026-09-29, reviewer): before any fresh-sample ablation cell is read, the v1 corner
  runs once on the v1 test items.
  - Its per-item predictions (and shortlists, where stored) must equal `bench_el_pg.json`'s `el_full` M_pg.
  - The result is recorded in the Record section.
  - Any difference means the corner is not v1, and it is fixed or disclosed before the ablation is read.
- **Wording:** the vectors factor also changes the embedded text: the KILT `title + '. ' + abstract` per page, against
  MahaBodi's per-passage text. The write-up calls it "the v1 vector layout", not "one vs many vectors".

## Clarification 3 (2026-09-29 EDT, before step 2 runs): the vector index at 5.9M may have to be IVFFlat

- **Problem:** at 5.9M pages there are ~23M passage vectors (384 dims, ~35 GB as float4). The v1 HNSW over 5.9M page
  vectors took 32,983 s (9.2 h) on the Mac mini's 16 GB (`bench_el_pg` run log). An HNSW over ~4× as many vectors,
  far larger than RAM, is projected at 40+ h and may not finish.
- **Step 2 therefore measures two index types on dev** (the dev 100K store, 500 dev mentions, dense recall@50 against
  exact search):
  - **HNSW:** m = 16, ef_construction = 64, `ef_search` ∈ {40, 100, 200, 400};
  - **IVFFlat:** `lists` = 4·√N (N = passages in the store), `probes` ∈ {10, 20, 40, 80}.
- **Rule, fixed now:**
  1. First, time the HNSW build on dev. Project it to 5.9M as dev time × (23M / N_dev) × log₂(23M) / log₂(N_dev).
  2. If the projection is ≤ 48 h, use HNSW with the smallest `ef_search` whose recall@50 ≥ 0.98 (or 400).
  3. Otherwise use IVFFlat with the smallest `probes` whose recall@50 ≥ 0.98 (or 80).
  4. If the chosen index can't be built at 5.9M within 72 h, or runs out of memory, a dated clarification says so
     before any 5.9M query is scored.
- **Reported:** index type, build time, recall and latency. The bridge on the test 100K runs with the same index type
  and setting.

## Clarification 3a (2026-09-29 EDT, before step 2 selects anything; supersedes clarification 3's rule)

Clarification 3's rule is replaced, after review:

- The 100K build-time projection crosses a memory regime: 100K fits in `maintenance_work_mem`, but 23M vectors on
  16 GB do not.
- Search settings chosen at 100K don't transfer to 23M: IVFFlat with `lists` = 4√N scans 6 % of lists at 100K and
  0.4 % at 23M; HNSW recall at a fixed `ef_search` also falls as N grows.

1. **Storage: halfvec (fp16) if it costs no recall.**
   - pgvector 0.8.0 `halfvec(384)` halves the vectors to ~17 GB at 23M.
   - Checked first on the dev 100K store: exact top-50 with halfvec vs float4 for the 500 dev mention queries.
   - halfvec is used at 5.9M if the mean overlap@50 is ≥ 0.99; otherwise float4 stays, and that is recorded.
2. **Index at 5.9M: IVFFlat** with `lists` = ⌈√rows⌉, pgvector's guidance for more than 1M rows (~4,800 lists at
   23M).
   - HNSW is not attempted at 5.9M. The only same-machine datapoint (v1: 5.9M float4 page vectors, 32,983 s at
     `maintenance_work_mem` 4 GB) already spilled, and 23M vectors are ~4× more data with less headroom. Its expected
     build time is well over 48 h.
   - HNSW and IVFFlat numbers at 100K are reported descriptively only.
   - `maintenance_work_mem` = 10 GB for the build, recorded with the build time.
3. **Search setting chosen at FULL scale on DEV queries:**
   - after the 5.9M store and its IVFFlat index are built, the 500 **dev** mention queries (no test item) run
     against it;
   - exact dense top-50 for those queries comes from a streaming NumPy scan over all stored passage vectors;
   - `probes` is the smallest value in {10, 20, 40, 80, 160, 320} whose mean recall@50 vs exact is ≥ 0.98, or 320
     if none reaches it (reported as such);
   - recall and latency are reported for every value.
4. **Bridge:** the test 100K bridge uses IVFFlat with ⌈√rows⌉ lists, and `probes` chosen the same way on the dev 100K
   store (dev queries, recall@50 ≥ 0.98), so the bridge also runs an approximate index of the same family.
5. **Limits:**
   - if the IVFFlat build at 5.9M exceeds 72 h or runs out of memory, a dated clarification says so before any
     5.9M query is scored;
   - so does a setting that can't reach 0.98 recall.
