# Pre-registration: alias prior + context decider on unseen entity-linking sets

Written 2026-09-28 EDT, before either test set has been downloaded to the compute machine, read or scored by any
script in this repository. The two KILT files are named only in `research/results/kilt_manifests/SHA256SUMS`.

## Question

On AIDA (PREREG_MILLION_SCALE.md), a context-free alias prior (P0: the most frequent entity for the mention string in
AIDA-train) beat every context-reading arm at every stage, by 40 or more points. Two questions follow.

1. Can a prior plus a context decider match or beat the prior alone, by reading context only where the prior is
   uncertain?
2. Does the decider matter there, MahaBodi `decide` against Laya `predict` on identical candidates and priors?

## Data

- **Test sets:** KILT `wned-dev-kilt.jsonl` (WNED-WIKI) and `cweb-dev-kilt.jsonl` (ClueWeb). Both are out of domain for
  AIDA and have never been used here. They are verified against the manifest SHA-256 before use (any mismatch stops the
  run).
- **Sample:** seed 0, 1,000 mentions per set, drawn after dropping items whose gold page is not among the 5,903,530
  KILT pages (the count dropped is reported). If a set has fewer usable items, all of them are used.
- **Mention and context:** extracted exactly as for AIDA (`kilt_el_prep.state_of`, the `[START_ENT] ... [END_ENT]`
  markers).
- **Candidate pool:** all 5,903,530 KILT pages.
- **Alias table:** mention → entity counts from `aidayago2-train-kilt.jsonl`, the same table as P0. For tuning on AIDA
  dev, the 500 dev mentions are excluded from it, as before.
- **Tuning set:** the existing 500 AIDA dev mentions only. No test item is looked at before the single test run.

## Arms (identical candidates for every arm except L1)

- **Candidates C(m):** the alias-table entities for the mention in count order, up to k = 20. If the mention is not in
  the table, or has fewer than 2 entries, the list is filled up to k = 20 from the dense MiniLM top-k over all pages
  (the same exact index as bench_el), skipping duplicates.
- **Prior share s(e):** count(m, e) / Σ count(m, ·). Entities that come from the dense fill get s = 0.
- **P0 (control):** the top alias entity, else the exact normalised title match, else no answer, exactly as in
  bench_el.
- **AP (primary: prior + MahaBodi decider):**
  - If max s ≥ τ, answer the top alias entity.
  - Otherwise run MahaBodi `decide` over C(m) with the context state, with options in seeded shuffled order
    (crc32(mention id)). Answer argmax over e of log p(e) + λ · log(s(e) + 0.01).
- **APL (control, same pipeline with Laya `predict`):** τ and λ are selected separately on dev by the same rule.
- **CTX:** MahaBodi `decide` over C(m), with τ = never and λ = 0 (context only).
- **L1:** Laya on the dense top-20 over all pages, with the bench_el settings. This is the reference from the AIDA run.
- **Options:** page title as the key; the description is "title: first n tokens of the abstract", with n = 48 as
  selected for M in bench_el.

The prior-gated combination is harness code (`research/bench_alias_prior.py`), not an engine feature. Results
describe it as "alias prior + MahaBodi decide", never as MahaBodi alone. It becomes a product feature only if it
passes here, and then only with a new pre-registered test.

## Selection (AIDA dev only)

- **Grid:** τ ∈ {0.5, 0.7, 0.9, never}, λ ∈ {0, 0.5, 1, 2}.
- **Rule:** maximise dev accuracy. Ties go to the higher τ, then the smaller λ (the more prior-like setting).
- **Scope:** selected once for AP and once for APL, then frozen. The per-cell dev table is committed and pushed before
  any test item is read.

## Metrics and verdicts (per test set; no pooling)

- **Reported per arm:** accuracy with a Wilson 95% CI, the answered count, and candidate recall (gold ∈ C(m)).
- **Primary:** AP vs P0 on each set, by exact McNemar. It is a beat only at p < 0.05; p ≥ 0.05 is a tie; a
  significantly lower AP is reported as a loss.
- **Secondary:** AP vs APL (decider only, same candidates and priors), AP vs CTX (value of the prior), and AP vs L1.
- **Subsets:**
  - "confident": max s ≥ 0.9;
  - "ambiguous": max s < 0.9;
  - "unseen": the mention is not in the alias table.
  Each is reported with n and accuracy per arm. They are descriptive only; no verdict is drawn from subsets.
- **Coverage disclosure:** the share of test mentions present in the AIDA-train alias table. Low coverage on these
  out-of-domain sets is expected and is reported, not fixed after the fact. A stronger table (for example Wikipedia
  anchor text) would need a new pre-registration and would be given to P0 as well.

## Machine and protocol

- **Machine:** the Mac mini used for the AIDA test, with CPU ONNX Runtime and 8 threads. The run starts only when no
  other benchmark is running.
- **Provenance:** every result file carries provenance: git revision, script hash and host.
- **Scoring:** the test runs once, with per-item predictions saved. The reviewing agent recomputes every number from
  the per-item files before anything is published.
- **Changes after this file is pushed:** only as dated clarifications written before test scoring.
