# Pre-registration: million-scale decision benchmarks

Written and agreed with the reviewer **before any of these benchmarks was run**. Changes after a
result is seen are allowed only as new, disclosed versions of this file; the original stays in git
history.

## Question

Can MahaBodi make typed decisions (choices and nouls) when each decision has millions of candidates
behind it? For example, one entity out of 5.9M Wikipedia pages, or a yes/no claim checked against
millions of passages. How does it compare with Laya given the same candidates?

## Common rules

- **One machine, same for every system:** the Ubuntu box (i9-9900X, 61 GB RAM, RTX 2080 Ti).
  Provenance goes in every result JSON (`research/provenance.py`).
- **Sample:** 1,000 decisions per benchmark, seeded shuffle (seed 0) of the test split.
- **Tuning:** every setting (shortlist size k, option text length, passage budget, gates) is tuned
  on train/dev splits only and fixed before the test sample is scored.
- **Statistics:** exact McNemar on the same items, and Wilson 95 % CIs.
- **Win rule:** M vs L1 at p < 0.05, **per stage**. Each stage's verdict is reported separately,
  with no pooled headline across stages. Losses are reported as losses.
- **Per-item predictions** are saved for every arm so the reviewer can recompute every number.
- **Publication:** results go at the top of README/BENCHMARKS only after the reviewer has
  recomputed them from the per-item files. Wording is neutral and puts the numbers, with the fair
  baseline, in the same line.

## Stages (candidate pool size)

The stages are 10K, 100K and the full pool (~5.9M for KILT), all on the same decisions.
- Each smaller pool is **gold + the top-H hard negatives** from the full pool (H = 100: the union
  of BM25 and dense neighbours of the mention or claim), **plus a seeded random fill**. The pools
  are nested: 10K ⊂ 100K ⊂ full.
- The hard-negative share is reported.
- The full stage is the natural pool.
- Every stage is reported, including any where a system breaks.

## Systems (arms)

| Arm | What it is |
|---|---|
| L0 | Laya alone, all candidates as options. It cannot run at ≥ 10K options (context limit), so it is reported as **"cannot run"**, never as 0 %. |
| L1 | Laya + MiniLM dense shortlist (top-k from the same pool); Laya decides. **The fair baseline; the headline compares against it.** |
| L1' | Laya on **M's** shortlist. This separates the retriever effect from the decision (tournament) effect. |
| D | Dense MiniLM top-1 |
| BM25 | BM25 top-1 |
| M | MahaBodi: memory holds the pool, retrieval gives a shortlist, and tournament + Laya decide. Optional, and reported as its own row: `learn()` on the train split, the handoff/OOS gate. |
| K | Plain kNN over the **train split's** labelled examples, with its own k and temperature tuned on dev (entity linking: the nearest training mentions vote for their entities). **Required whenever M uses `learn()`**, because the `learn()` row uses the same training data; its verdict vs K is reported next to its verdict vs L1. |

- Each of L1 and M gets **its own best k**, tuned on dev. Both are reported.
- **Decomposition:** recall@k of each shortlist (gold in the top-k), and decision accuracy given
  that gold is in the shortlist.
- **Option text:** title + first N tokens, with N fixed on dev and **the same for L1, L1' and M**.
  Truncation statistics are recorded.
- **Shared index:** any approximate-nearest-neighbour index (FAISS/HNSW) is shared by every arm that
  retrieves, so it is not an M-only advantage.

## Benchmarks

1. **Entity linking**: KILT AIDA-CoNLL (and WNED-WIKI, WNED-CWEB). Choose one Wikipedia entity out
   of ~5.9M.
   - State: the mention in its context. Options: entity title + first N tokens.
   - Metric: accuracy@1.
   - Test: KILT dev splits (test labels are hidden). Tuning: AIDA train.
2. **Fact verification**: KILT FEVER, a yes/no noul (SUPPORTS vs REFUTES; KILT excludes NEI).
   - Evidence is retrieved from ~5.9M Wikipedia pages.
   - Test: FEVER dev. Tuning: FEVER train.
   - Arms: Laya question-only; Laya + BM25 passages; Laya + dense passages; M
     (`decide_with_memory`). All retrieval arms get the same k and passage budget.
   - Also reported: gold-evidence retrieval recall, always-SUPPORTS accuracy, and published KILT
     state of the art, for context.
3. **Extreme classification**: WikiLSHTC-325K or Amazon-670K (standard split).
   - Standard **multi-label P@1, P@3 and P@5** on the full test split, with no filtering to
     single-label documents.
   - Baseline: kNN over training documents' labels. M: `learn()` on train.
   - Published P@1 is cited for context.
4. **Throughput** (not accuracy): 1M nouls and 1M choices end to end.
   - The headline figure is with the **cache OFF**, with a check that the inputs contain no
     duplicates. The cache-on figure is reported separately, with its hit rate.
   - Measured: decisions/s, p50/p95 per decision, peak RSS.
   - Systems: M, Laya PyTorch and Laya ONNX, each with its batch size and thread count stated.

## Scale path

MahaBodi memory has only been tested up to 2,000 paragraphs.
- If in-process memory cannot hold 5.9M items, the **PostgreSQL + Apache AGE + pgvector** path
  (Enterprise.md P4) is used instead, and the result records which path ran. This doubles as a real
  test of the enterprise design.
- `learn()`'s leave-one-out cost is O(min(n, 2000) · n). Its time at AIDA-train and FEVER-train size
  is measured and reported.

## Context and third parties

- Published state-of-the-art numbers (e.g. BLINK/GENRE on AIDA, KILT FEVER leaders, XMC P@1) are
  cited for context. If MahaBodi is far below them, the README says so.
- **TypeSafe Jev** (the closed decision API that Laya's README benchmarks against) has never been
  run in this project.
  - Its **published** numbers (Jev 1.13.0; AbdelStark/jev-benchmarks,
    nibzard/decision-model-benchmark) may be cited only on the same datasets, labelled as
    third-party.
  - It is never presented as a head-to-head we ran, unless we get API access and run it on our
    items.

## Data

- KILT files come from `dl.fbaipublicfiles.com/KILT`: knowledge source and the AIDA, WNED, CWEB
  and FEVER splits.
- They are stored on Ubuntu in `/media/sda/data/kilt`, with a SHA256SUMS file recorded.

## Addendum 1 (2026-09-26, before any run): option-count scaling curve

Agreed with the reviewer before anything is measured. It answers one question: how accuracy and latency change as
the number of options per decision grows, from small (where Laya was built to work) to millions.

- **Data:** entity linking on the same seeded 1,000 KILT AIDA dev mentions as above.
- **Option counts:** N ∈ {4, 20, 77, 150, 1K, 10K, 100K, full ~5.9M}. Each pool is gold + hard negatives + seeded
  random fill, nested as above.
- **Option text:** title + first N tokens, fixed on dev, the same for all arms at every N.
- **Arms at every N:**
  - L0-PyTorch: Laya as shipped, all N options in one call.
  - L0-ONNX: the Laya-identical ONNX path (`predict`, no tournament, no memory).
  - L1: MiniLM shortlist → Laya.
  - M: MahaBodi.
  - D: dense top-1.
- **Latency:** the headline latency comparison is **M vs L0-ONNX (same runtime)**, with L0-PyTorch shown alongside,
  so a runtime difference is never reported as a method win.
- **Metrics per arm per N:**
  - accuracy@1 with a Wilson CI; exact McNemar M vs L0 and M vs L1;
  - p50/p95 latency per decision (CPU, same threads, batch 1, cache off);
  - peak RSS;
  - index/memory build time per N, reported separately from per-decision latency, as an amortised cost.
- **Decomposition** for M and L1 at every N: shortlist recall@k (gold in the shortlist), and accuracy given the gold
  is in the shortlist.
- **L0 degradation:**
  - Per N, record the fraction of options actually visible to Laya after its own packing/truncation.
  - Any truncation = **degraded**: accuracy is still reported, but flagged.
  - **"Cannot run"** only for out-of-memory, a 60 s/decision timeout, or zero visible options, with the reason and N.
  - Never shown as 0 %.
- **Win rule:** per N (p < 0.05), with no pooled headline.
- **Wording is bound to the data.** Words like "crawls" are not allowed. Allowed forms:
  - "Laya cannot run beyond N = … (reason)";
  - "at N = …, M takes X ms vs L0-ONNX Y ms";
  - accuracy with CIs.
- **Both ends of the curve** (including any N where M is slower or less accurate) go in the same figure/table.

## Pre-run clarification 1 (2026-09-27, before any pool is built or arm scored): entity-linking pools

Agreed with the reviewer.

- **One shared pool per stage, the same for every arm.** MahaBodi's memory holds one pool, so per-mention pools of
  10K–100K pages for 1,000 mentions are not feasible.
- **Pool contents** at stage N:
  1. The golds of the 1,000 test mentions.
  2. Per mention, the top-h hard negatives from the full 5.9M pages: the dense and BM25 top-100 lists interleaved by
     rank, gold excluded, with h = min(100, ⌊0.5·(N − G)/M⌋). Hard negatives take at most half the pool.
  3. A seeded random fill from the remaining pages, up to N.
- **Nesting:** 10K ⊂ 100K ⊂ full. The full stage is the natural 5.9M.
- **Reported per stage:** h and the hard-negative share. The dev set (500 AIDA-train mentions, used only to tune k and
  the option text length) gets its own separate pools.
- **Mention selection:** mentions whose gold page is not in the KILT page table are dropped before sampling, and the
  count is reported (`el/MANIFEST.json`).
- **State:** the mention with ±200 characters of context, [START_ENT]/[END_ENT] markers kept.
- **Pool-bias control (arm P0), reported at every stage:** context-free baselines that never read the context.
  - (a) exact/normalised title match of the mention string, with ties broken by (b);
  - (b) a popularity prior: the entity most often linked from that mention string in AIDA train, falling back to
    (a).
  - Any stage where P0 ≥ L1 is flagged **"pool-biased"**, and its verdict is not a headline.
- **Headline stage:** the natural full 5.9M pool only. The 10K/100K stages are the diagnostic scaling curve, each shown
  with its hard-negative share and P0.
- **Near-duplicates:** per stage, the fraction of hard negatives whose normalised title equals the gold's (these are the
  hardest, and the likeliest to carry gold-label noise), plus a random 20 listed for manual inspection.
- **Test/dev disjointness:** test and dev gold page ids are kept out of each other's pools where possible, and any
  overlap is reported.

## Pre-run clarification 2 (2026-09-27, before any test arm is scored): retrieve by mention, decide with context

Agreed with the reviewer, based on the dev-only table `results/el_query_forms_dev.json` (500 dev mentions, recall over
the full 5.9M pages). Recall is shown as @1 / @10 / @100:

| Query | Dense | BM25 | Either, @100 |
|---|---|---|---|
| mention string only | 0.054 / 0.310 / 0.674 | 0.054 / 0.214 / 0.440 | 0.774 |
| mention ±50 characters | 0.034 / 0.112 / 0.196 | 0.048 / 0.154 / 0.306 | 0.362 |
| mention ±200 characters (the previous state) | 0.040 / 0.062 / 0.166 | 0.038 / 0.086 / 0.210 | 0.258 |

The changes:

- **Retrieval uses the mention string.** That covers D, BM25, the L1 shortlist and MahaBodi's `query` in M.
- **Every decision sees the ±200-character context state** (Laya in L1/L1', MahaBodi `decide` in M). This is the
  standard retrieve-then-rerank layout for entity linking. P0 is unchanged (context-free), and so is K (context
  kNN).
- **Hard negatives are re-mined with the mention query** (dense ∪ BM25, interleaved) and the pools rebuilt with the same h
  rule and seeds. The ctx200-mined pools stay on disk as a disclosed prior version, not used. The change in
  hard-negative share is reported.
- **No mention→entity alias table is added** to the candidate sources. P0(b) (the AIDA-train mention→entity prior) is
  the context-free control that shows how much the context-reading arms add over alias knowledge.
- **Ceiling disclosure:** every shortlist arm (L1, L1', M) is capped by its shortlist recall (union @100 = 0.774 on
  dev). Shortlist recall@k and accuracy-given-gold-in-shortlist are reported per stage and arm.
  - Published state-of-the-art entity linkers (~0.9 on AIDA) use alias/anchor-text candidate tables, which none of our
    arms uses.
- **The query form is fixed from the dev table above** and is not revisited after test.

## Pre-run clarification 3 (2026-09-27, before dev tuning finished and before any test arm is scored): selection rule

- **Selection:** for each of L1 and M, (k, n) is chosen as the **highest dev accuracy** on the dev 100K pool, with ties
  going to the smaller k, then the smaller n. K's (k, T) uses the same rule over its grid, with ties to the smaller k.
  The grids are k ∈ {5, 10, 20, 50}, n ∈ {16, 48}; K: k ∈ {1, 5, 10, 25}, T ∈ {0.02, 0.05, 0.1}.
- **k and n are selected on dev at 100K and applied unchanged at every stage** (10K, 100K, full 5.9M). There is no
  re-tuning per stage, and none after seeing any test result.
- **Reported as-is:** the dev k-curve (accuracy, shortlist recall and accuracy given gold-in-shortlist, per (k, n), for
  L1 and M), with per-item predictions and shortlist hits in `bench_el_tune.json`.
- **L1 and M have different shortlists:** L1's is the dense MiniLM top-k within the pool; M's is MahaBodi's hybrid
  `query` top-k pages. Each arm's shortlist hits are stored separately.
- **Clarification 3a (2026-09-27, before any test arm is scored):** L1' uses **M's selected (k, n)** and the
  **identical per-item shortlist** that M's `query` returns. Only the decider differs: Laya `predict` for L1', MahaBodi
  `decide` for M. So L1' vs M is a controlled comparison of the decision step. The retrieval query falls back to the
  context state only when a mention string is empty; this happens for 0 of 500 dev and 0 of 1,000 test mentions.

## Pre-run clarification 4 (2026-09-27, before dev tuning finished and before any test arm is scored): alias-candidate arm, ablations, full-stage M

- **Why:** clarification 2 kept M and L1' free of any mention→entity alias table, and those arms stay exactly as
  specified. The dev table shows every shortlist arm capped by shortlist recall, so one **additional** arm tests whether a
  MahaBodi memory of aliases improves candidate generation. It is labelled as an addition and replaces nothing.
- **MA (alias candidates + context decider):**
  - One MahaBodi memory record per distinct (mention string, gold entity) pair in `aidayago2-train-kilt.jsonl`, with the
    500 dev-sample mentions excluded. This is the same source as P0. No testa/testb strings are used.
  - The record text is `# <mention>\n\nrefers to: <entity title>` and contains **no counts or frequencies**.
  - MA's shortlist is the alias-memory `query(mention)` hits (in pool, de-duplicated, in rank order), followed by M's page
    shortlist, truncated to k.
  - The decider is MahaBodi `decide` over the context state with the entity pages as options, as for M. The decider never
    sees the alias records.
  - MA uses **M's selected (k, n)**. Nothing about MA is tuned, and it is not run on dev before test.
- **L1'-MA:** Laya `predict` on MA's identical per-item shortlist. This is the controlled decision-step comparison for MA,
  as L1' is for M.
- **Option order:** options are presented in retrieval rank order in every arm. Two seeded-shuffle ablations are
  reported as disclosed diagnostics, not claims: `MA_shuffled` and `M_shuffled`. Each uses the same shortlist, with the
  order shuffled by `random.Random(crc32(mention_id))`.
- **Reported for MA:** accuracy with a Wilson CI, shortlist recall, accuracy given gold-in-shortlist, p50 latency, exact
  McNemar vs L1 and vs M, and per-item predictions.
- **Full 5.9M stage for M, L1', MA and L1'-MA:**
  - The in-process memory cannot hold 5.9M pages (probe_memory_scale), so these arms run through the PostgreSQL path at
    the full stage.
  - **No full-stage test result is scored for any arm until the PG path runs.** The dense, BM25, P0, K and L1 full-stage
    numbers are computed in the same run, not earlier.
  - If the PG path cannot run, a separate dated clarification will say why before anything at the full stage is
    reported. Losses and non-runs are reported as such.
- **Dev grid noise:** with n = 500 dev mentions (Wilson CI ≈ ±0.035), differences within the L1 and M (k, n) grids that
  are smaller than that are within noise. The selection rule (clarification 3) still decides. The results state that the
  selected setting is not significantly better than its neighbours wherever that holds.

## Pre-run clarification 4b (2026-09-27, before the PostgreSQL store is loaded and before any test arm is scored): the PG path, defined

- **Why:** `decide` is a pure function of (state, options). It does not read memory
  (`engine.rs`: only `decide_with_memory` does), so at 5.9M pages only **candidate generation** needs the store.
- **The PG path:** the reference store in `deploy/postgres` (PostgreSQL 17, pgvector, AGE image from
  `deploy/postgres/Dockerfile`) with the retrieval in `deploy/sync/mahabodi_pg.py::search`.
  - The ranking is unchanged from that file: lexical (`tsvector`, OR-of-terms) plus pg_trgm typo correction plus dense
    pgvector (cosine), fused by reciprocal rank (rrf_k = 60, pool = 50).
  - It is queried by the mention string, as M is (clarification 2).
- **The rows are what MahaBodi's own snapshot holds for these pages.** A snapshot of `# <title>\n\n<abstract>` has
  action `Passage` and body `<title>: <abstract>`.
  - The loader writes those fields directly with COPY, with id `pg_<row>`.
  - The AGE graph is not loaded, because `search` does not read it. This is stated in the results.
  - A 10K-page check compares the loader's bodies with `Bodi.snapshot()` bodies before the full load. Any mismatch stops
    the run.
- **Dense vectors:** the existing KILT MiniLM vectors (text `title + '. ' + abstract`; parity with the MahaBodi embedder
  min cos 0.99999988), in an HNSW index (m = 16, ef_construction = 64, `hnsw.ef_search` = 100). The query vector comes
  from the MahaBodi embedder.
- **Arms at the full stage:**
  - **M_pg** = PG shortlist → MahaBodi `decide`;
  - **L1'_pg** = the same shortlist → Laya `predict`;
  - **MA_pg** = alias-memory hits, then the PG shortlist → `decide`;
  - **L1'-MA_pg** = MA_pg's shortlist → Laya `predict`.
  - All use M's selected (k, n), with no tuning.
- **Bridge (same run, test 100K pool):** the same PG path is also scored over a namespace that holds exactly the test
  100K pool. M_pg vs the in-process M on the same pool measures what the PG retriever changes, and is reported with
  exact McNemar.
  - Full-stage M_pg is **not** described as the in-process M. The results label it "MahaBodi via PG store".
- **Stability:** loads and HNSW builds are resumable, and query runs checkpoint per mention. The same rules as
  clarification 4 apply to non-runs.

## Pre-run clarification 4c (2026-09-27, before any row is loaded into the store and before any test arm is scored): passages, not pages

- **The 4b parity check failed as designed and stopped the run:** 8,835 of the first 10,000 pages differ between the
  one-row-per-page loader and `Bodi.snapshot()`.
  - Cause: MahaBodi splits a page into sentence-group passages. On the first 2,000 pages it made 10,691 ATFs (1–10 per
    page, median 6). Only the first passage carries the `<title>: ` prefix.
  - The check output is kept in `/media/sda/pg_el/state/check.json`.
- **The rows are therefore the snapshot itself.** Pages are ingested into an in-process `Bodi` in 50K-page batches
  (without an embedder), `snapshot()` is taken, and its ATFs (id, action, data_connections, body) are copied into the
  store unchanged.
  - A page is ingested exactly as M ingests it (`# <title>\n\n<abstract>`, source `pg<row>`).
  - About 31M passages are expected for 5.9M pages.
- **Dense vectors:** the existing page vector (`title + '. ' + abstract`) is attached to the page's **first passage
  only**. In-process M embeds every passage with the MahaBodi embedder, so the PG dense stage is coarser than M's.
  Re-embedding ~31M passages on this machine's CPU is not feasible in this run. The 100K bridge measures the combined
  effect of this and the SQL ranking.
- **Page shortlist:** `search(k = 3k)` passages are mapped to pages (`pg_<row>_…`) in rank order, de-duplicated and
  truncated to k. This is the same rule as in-process `m_short`.
- Everything else in 4b stands (HNSW parameters, arms, bridge, labels).
