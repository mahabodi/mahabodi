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
