# Pre-registration: alias prior + context decider on unseen entity-linking sets

Written 2026-09-28 EDT, before either test set has been downloaded to the compute machine, read or scored by any
script in this repository. The two KILT files are named only in `research/results/kilt_manifests/SHA256SUMS`.
Revised the same day after review, still before any data was read (5ae7412 was the first draft): a P0L control,
in-set tuning splits, a probability floor, the dense-fill specification, a leakage statement and extra
disclosures.

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
- **Sample and split:** items whose gold page is not among the 5,903,530 KILT pages are dropped, and the count is
  reported. Per set, seed 0 shuffles the rest.
  - The first 200 are that set's **tuning split**.
  - The next up to 1,000 are its **test split**.
  - Tuning happens in-set because the alias table covers AIDA (~95%) far better than these sets, so settings chosen
    on AIDA may not transfer.
- **Mention and context:** extracted exactly as for AIDA (`kilt_el_prep.state_of`, the `[START_ENT] ... [END_ENT]`
  markers).
- **Candidate pool:** all 5,903,530 KILT pages.
- **Alias table:** mention → entity counts from `aidayago2-train-kilt.jsonl`, the same table as P0. For tuning on AIDA
  dev, the 500 dev mentions are excluded from it, as before.
- **Tuning:** on each set's 200-item tuning split only. The AIDA dev grid is also run and reported, descriptively. No
  test-split item is read before the single test run.

## Arms (identical candidates for every arm except L1)

- **Candidates C(m):** the alias-table entities for the mention in count order, up to k = 20. If the mention is not in
  the table, or has fewer than 2 entries, the list is filled up to k = 20 from the dense MiniLM top-k over all pages.
  - The fill uses the exact inner-product index of bench_el, queried by the mention string (clarification 2 of
    PREREG_MILLION_SCALE).
  - Alias entities already in the list are skipped.
- **Prior share s(e):** count(m, e) / Σ count(m, ·). Entities that come from the dense fill get s = 0.
- **P0:** the top alias entity, else the exact normalised title match, else no answer, exactly as in bench_el.
- **P0L (primary control):** P0, else L1's answer wherever P0 abstains. Out of domain, P0 will abstain often, and AP
  must not "beat" the prior merely through its fallback.
- **AP (primary: prior + MahaBodi decider):**
  - If max s ≥ τ, answer the top alias entity.
  - Otherwise run MahaBodi `decide` over C(m) with the context state, with options in seeded shuffled order
    (crc32(mention id)). Answer argmax over e of log max(p(e), 1e-6) + λ · log(s(e) + 0.01).
  - `decide` uses the same defaults as M in bench_el, including the tournament. Options the tournament eliminates
    get p = 0 (decide.rs), so the floor lets a strong prior still rescue them.
  - APL uses the same floor.
- **APL (control, same pipeline with Laya `predict`):** τ and λ are selected separately on dev by the same rule.
- **CTX:** MahaBodi `decide` over C(m), with τ = never and λ = 0 (context only).
- **L1:** Laya on the dense top-20 over all pages, with the bench_el settings. This is the reference from the AIDA run.
- **Options:** page title as the key; the description is "title: first n tokens of the abstract", with n = 48 as
  selected for M in bench_el. With 20 options, Laya's packing truncates each option to its per-option share of the
  head budget, so the effective length is shorter than 48 tokens. This is the same for every decider.

The prior-gated combination is harness code (`research/bench_alias_prior.py`), not an engine feature. Results
describe it as "alias prior + MahaBodi decide", never as MahaBodi alone. It becomes a product feature only if it
passes here, and then only with a new pre-registered test.

## Selection (each set's tuning split)

- **Grid:** τ ∈ {0.5, 0.7, 0.9, never}, λ ∈ {0, 0.5, 1, 2}.
- **Rule:** maximise tuning-split accuracy. Ties go to the higher τ, then the smaller λ (the more prior-like
  setting).
- **Scope:** selected separately for AP and APL and for each set, then frozen. The per-cell tuning tables are
  committed and pushed before any test-split item is read.

## Metrics and verdicts (per test set; no pooling)

- **Reported per arm:** accuracy with a Wilson 95% CI, the answered count, and candidate recall (gold ∈ C(m)).
- **Primary:** AP vs P0L on each set, by exact McNemar. It is a beat only at p < 0.05; p ≥ 0.05 is a tie; a
  significantly lower AP is reported as a loss.
- **Secondary:**
  - AP vs P0;
  - AP vs APL (decider only, same candidates and priors);
  - AP vs CTX (value of the prior);
  - AP vs L1;
  - P0L vs L1 (what the prior alone adds out of domain).
- **Subsets:**
  - "confident": max s ≥ 0.9;
  - "ambiguous": max s < 0.9;
  - "unseen": the mention is not in the alias table.
  Each is reported with n and accuracy per arm. They are descriptive only; no verdict is drawn from subsets.
- **Candidate provenance:** per set, the share of items where gold ∈ C(m) came only from the dense fill.
- **Coverage disclosure:** the share of test mentions present in the AIDA-train alias table. Low coverage on these
  out-of-domain sets is expected and is reported, not fixed after the fact. A stronger table (for example Wikipedia
  anchor text) would need a new pre-registration and would be given to P0 as well.

## Leakage

- Laya's README does not publish its full training data. It names AG News and BoolQ as "in training mix"
  (third_party/laya/README.md lines 675–676) and says nothing about KILT or entity-linking data.
- The MiniLM embedder's training data is not verified here.
- So leakage of WNED-WIKI or ClueWeb into either model cannot be ruled out. This is stated with the results. It
  affects every arm that uses those models, P0 excepted.

## Machine and protocol

- **Machine:** the Mac mini used for the AIDA test, with CPU ONNX Runtime and 8 threads. The run starts only when no
  other benchmark is running.
- **Provenance:** every result file carries provenance: git revision, script hash and host.
- **Scoring:** the test runs once, with per-item predictions saved. The reviewing agent recomputes every number from
  the per-item files before anything is published.
- **Changes after this file is pushed:** only as dated clarifications written before test scoring.

## Clarification 1 (2026-09-28 EDT, after the data was downloaded and checksum-verified, before any item was tuned or scored)

- **The contradiction:** the selection rule said "ties go to the higher τ, then the smaller λ (the more prior-like
  setting)". Those two directions are the *less* prior-like ones. A higher τ fires the prior gate less often, and a
  smaller λ weights the prior less.
- **Resolution:** the stated intent, the more prior-like setting, is kept. Ties go to the **lower τ** (τ = never
  counts as the highest), then the **larger λ**.
- **Why:** this is the conservative choice, because it leans towards the control that won on AIDA.
- **Where it's implemented:** `research/bench_alias_prior.py` (selection `rank`).
- **Data read so far:** the downloaded files were only checksum-verified. No item has been read.
