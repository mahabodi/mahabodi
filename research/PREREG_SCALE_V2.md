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
- **The 5.9M run:** the same 1,000 test mentions as bench_el.
- **2 × 2 descriptive ablation:** per-passage vectors on/off × lexical cascade on/off. "Cascade on" means the
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
