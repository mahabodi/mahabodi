# Pre-registration: MahaBodi as an agent router, against hosted LLMs

Written 2026-09-30, before any item below has been scored by any arm named here. Revised the same day after review,
still before any data was read. Changes after this file is pushed are made only as dated clarifications, written
before test scoring.

## Question

For high-volume routing (choosing one of 60–151 intents, queues or tools per request), how does MahaBodi's accuracy
compare with a hosted LLM's, and at what cost and latency? Two conditions are tested:

- **Zero-shot:** only the label names are available.
- **Labelled:** training examples are available too.

The buyer's alternative is an LLM call per request, or a cheap embedding classifier. It isn't Laya.

**Expectation, stated in advance:** on CLINC150 zero-shot, a current LLM with the full label list will very likely
be more than 3 points more accurate than MahaBodi (0.74 in earlier runs). The pre-registration is designed so that
this shows if it happens. The candidate claim may be the labelled, cheap and fast condition, not zero-shot.

## Suites and items

| Suite | Options | Source | Test items |
|---|---|---|---|
| CLINC150 "plus" | 150 intents + out-of-scope | `clinc_oos` / `plus`, test split | 1,000 |
| Banking77 | 77 intents | `PolyAI/banking77`, test split | 1,000 |
| MASSIVE en-US | 60 intents | `AmazonScience/massive` `en-US`, test split | 1,000 |

- **Fresh items:**
  - per suite, every test item already scored by an earlier MahaBodi run (ids in `research/results/*.json`) is
    excluded;
  - `random.Random(20260930)` then shuffles the rest in file order and takes the first 1,000;
  - the id lists and their SHA-256 are committed before any arm runs.
- **Validation:** 200 items per suite from the validation split (Banking77 has none, so a seeded sample of `train`,
  disjoint from the labelled pool below).
  - It is used only to check that the LLM request and answer format work.
  - No prompt wording is tuned for accuracy.
  - No test item is read before the single test run.
- **Labelled pool:** each suite's training split, minus the Banking77 validation sample. The same pool serves every
  labelled arm.
- **Misspelled variant:** each test item also runs with every word of 5 or more letters losing one interior letter,
  seeded by item index (`research/bench_retrieval.py`, `typo_all`). Clean and misspelled are reported separately.

## Arms

### Condition Z: zero-shot (label names only)

- **M (MahaBodi):** `decide` with shipped defaults (tournament), options = the intent names, decision cache off.
- **L (Claude Haiku 4.5, `claude-haiku-4-5-20251001`)** and **L2 (Claude Sonnet 5, `claude-sonnet-5`):**
  - temperature 0;
  - one fixed prompt: the instruction, the label list, the utterance;
  - the answer is returned through tool use, with one parameter whose JSON-schema `enum` is exactly the label names,
    so every answer is a valid label. A call that returns no tool call counts as wrong and is reported separately as
    "invalid";
  - the label list and instruction form a fixed prefix, marked for prompt caching;
  - token usage (input, cached input, output) is recorded per call.
- **CLINC150 out-of-scope:**
  - M uses the shipped opt-in gate at the threshold fixed in the W11 re-test (`bench_clinc_oos.json`, tau 0.5,
    s 0.3). That threshold was tuned at the documented test prevalence of out-of-scope items (18.2 %), which is
    disclosed again here.
  - L and L2 get `out_of_scope` as an enum value, with an equally explicit instruction: "Choose out_of_scope if the
    request matches none of the listed intents."

### Condition S: labelled (training examples available)

- **ML (MahaBodi):** `learn(..., calibrate=200)` on the labelled pool, as shipped. `decide` with the cache off.
- **E (embedding kNN):** all-MiniLM-L6-v2 over the labelled pool, k = 10, majority vote, ties broken by summed
  similarity.
- **LF (Haiku 4.5, few-shot):**
  - the zero-shot prompt plus the k = 10 nearest labelled utterances (the same MiniLM neighbours as E), each with its
    label, placed after the cached prefix;
  - the same tool-use enum.

## Metrics

- **Accuracy:** Wilson 95 % CIs, and paired comparisons:
  - exact McNemar, with the raw discordant counts;
  - a 95 % CI on the paired accuracy difference (Newcombe's hybrid-score interval for paired proportions, method 10).
- **CLINC150:** overall accuracy (151 classes), in-scope accuracy and out-of-scope recall, per arm.
- **Latency per decision:**
  - M, ML and E on the Mac mini (M2 Pro, CPU, 8 threads), batch 1, over the scored pass in item order, with no
    warm-up (PREREG_SCALE_V2 clarification 3d). Reported: the first call, p50/p95 overall, and p50/p95 over the
    second half.
  - LLM arms end to end from the calling machine, network included. The machine, its network and the API region,
    and the date and time are recorded.
- **Cost per 1M decisions:**
  - **LLM arms:** recorded tokens × Anthropic's published price on the run date, quoted with its URL and date. Two
    figures are reported:
    - at list price;
    - with prompt caching of the fixed prefix, at the cached-input price.

    The 10× cost test uses the **cached** figure, which is the conservative choice for MahaBodi. Batch pricing is
    also reported, descriptively.
  - **M, ML and E:** a named on-demand cloud instance type, the smallest whose RAM holds the arm's measured peak RSS
    with 25 % headroom. Its hourly price (URL and date quoted) is divided by the arm's measured sustained throughput
    in decisions per hour, at the thread count used, over the scored pass.
    - Throughput is measured on the Mac mini. Mapping it to the cloud instance is an estimate, and is stated as one.
    - The same instance family is used for every arm run on CPU.

## Verdicts (per suite, per variant, per LLM; each cell stands alone, nothing pooled)

**Accuracy margin.** A router's error costs a misroute: a handoff or a retry, not a harmful answer. A 3-point
non-inferiority margin is therefore used for routing. It is fixed here and not changed after data.

- **Primary, condition Z:** M vs L, and M vs L2, each stated separately ("vs Haiku 4.5", "vs Sonnet 5").
  - **"Router win"** requires all three of:
    1. the lower 95 % bound of (M − L) is ≥ −0.03;
    2. cost per 1M decisions, against the LLM's cached price, is ≥ 10× lower;
    3. p50 latency is ≥ 10× lower.
  - A significant M > L on McNemar (p < 0.05) is also reported, as "beat on accuracy".
  - **Otherwise the result is reported as it is.** For example, "M is 9 points less accurate than L2 (a loss on
    accuracy), at 1/200 of the cost" is written exactly like that.
- **Primary, condition S:** ML vs LF and ML vs E, under the same rule. LF is Haiku only, so the labelled-condition
  verdict reads "vs Haiku few-shot". Sonnet few-shot is not tested. The E comparison has no cost or latency
  multiple: the verdict is accuracy only, with cost and latency reported alongside.
- **Secondary:** M vs E; E vs LF; L vs L2.

## Budget, for the user's approval before any LLM call

- **Calls:** per LLM arm, (1,000 test × 2 variants + 200 validation) × 3 suites = 6,600.
- **Input per call:**
  - about 450 tokens (MASSIVE), 550 (Banking77) or 900 (CLINC150): the label list, instruction and tool schema;
  - plus about 250 for LF's examples;
  - output about 30 tokens.
- **Arms:** L (Haiku, zero-shot); L2 (Sonnet 5, zero-shot); LF (Haiku, few-shot).
- **Estimated spend:** of the order of tens of US dollars in total at list prices, less with caching.
  - The exact figure is computed from the published prices on the run date and shown to the user.
  - No LLM call is made before the user approves it and provides the API key.
- **L2 cap option:** if the budget should be capped, L2 runs on the clean variant only. That option has to be chosen
  before any L2 call, in a dated clarification.

## Rules

- **One test run.** Per-item predictions, raw LLM responses and token counts are saved. The reviewing agent
  recomputes every number from them before anything is cited.
- **Same machine:** M, ML and E run on the Mac mini, only when no other benchmark is running (after the
  PREREG_SCALE_V2 chain). The LLM arms are remote APIs, and the calling machine is recorded.
- **Laya checkpoint:** `models/laya-v2` (English) for all three suites, which are all English. The multilingual
  checkpoint is not used.
- **No secret is stored in the repo or in any result file.** The API key is read from the environment at run time
  only.
- **Leakage:** Laya's training mix and the LLMs' training data may include these public datasets. This can't be
  ruled out, and is stated for every arm.
- **Losses are reported as losses,** in BENCHMARKS, in the README and on the site.

## Clarification 1 (2026-10-01, before any item is scored by any arm): LLM arms deferred

- **Why:** the user has not yet provided an API key, and asked to skip that part for now.
- **What runs first:** only the arms that need no API, on the pre-registered fresh items:
  - M (MahaBodi zero-shot);
  - ML (MahaBodi with `learn(calibrate=200)`);
  - E (MiniLM kNN).
- **What stays fixed:** the LLM arms' prompt, tool-use enum, model ids, items, budget rule and verdict rules are
  unchanged. They can't be influenced by seeing M, ML or E first, because nothing about them is chosen after this
  point.
- **What is reported until the LLM arms run:** only the secondary comparisons M vs E and ML vs E, labelled "LLM arms
  pending". No router claim against any LLM is made.
- **When the LLM arms run later:** they use the same items and the same per-item files, and the primary verdicts are
  computed then. The time between the two runs is stated.
- **Sonnet:** before any LLM call, a further dated clarification records the Sonnet model choice. Sonnet 5 is now
  listed as legacy; Sonnet 5.5 is current at the same price.

## Clarification 2 (2026-10-01, before items are built or any arm runs): details the harness had to fix

`research/bench_router.py`:

1. **Excluded test items.** No earlier result file stores item ids, so each earlier selection is re-derived from its
   seed and slice, and checked against that run's saved gold labels. The `items` phase stops on any mismatch.
   - **CLINC150:** positions 0–1999 of the test split shuffled with seed 7 (bench_clinc 0–999; W11 1000–1999). That
     excludes 2,000 of 5,500.
   - **Banking77:** positions 0–1999 of `mteb/banking77` test shuffled with seed 0 (bench, fresh_experience, fresh3,
     fresh4_head), plus bench_smoke's first 12 unshuffled rows. These are mapped to `PolyAI/banking77` rows by exact
     text, with the match counts recorded. About 1,070 items remain, so the test takes the first 1,000 of them.
   - **MASSIVE en-US:** the first 100 rows of `mteb/amazon_massive_intent` en, mapped to AmazonScience ids, with the
     texts checked.
2. **Options and format.** The same as the earlier published runs (bench_clinc arm C, bench.py, bench_massive), with
   one difference for MASSIVE: all 60 intents are offered, sorted (as this pre-registration says), not the earlier 20.
3. **CLINC150 out-of-scope in the labelled condition:**
   - `learn` accepts only labels that are options, so ML learns the in-scope training rows.
   - ML gets the same shipped W11 gate as M.
   - E treats `out_of_scope` as an ordinary class.
   - Each item's raw top-1, probability, similarity and gate flag are saved, so the result without the gate can be
     recomputed and reported.
4. **Validation sample:** `random.Random(20260930)`, the first 200, from the validation split (Banking77: train,
   disjoint from the pool). Validation items used by earlier runs are not excluded; this pre-registration requires
   that only for test items.
5. **Hashes and typos:** ids are strings, sorted as text, and the SHA-256 is taken over them joined by newlines. The
   misspelled variant is seeded by item position (0–999).
6. **Dataset source:** if a Hub dataset still needs a loading script, the script falls back to the Hub's parquet copy
   and records which it used.
7. **Latency and memory notes:**
   - E's first call is already warm (its embedder has just embedded the pool).
   - ML's learn and calibrate time is recorded separately from the scored pass.
   - Peak RSS covers the whole process, including the loaded datasets.

## Clarification 3 (2026-10-01, before items are built): audit of earlier item use

This is a review follow-up to clarification 2. Every `research/results/*.json` that mentions `banking77`, `clinc` or
`massive` (case-insensitive) is classified below. The table also covers the item-level files that don't match that
pattern: `bench_clinc.json`, `bench_clinc_oos_arrays.json`, `bench_massive.json` and the `check_oos_*.json` files.

- **Enforced in code:** the table is the `AUDIT` list in `research/bench_router.py`. `--phase items` enforces it
  before it builds anything:
  - it stops if any matching result file is missing from the list (`router_*` files excepted);
  - it asserts that every test range lies inside the excluded span of the same shuffle;
  - it runs each file's recorded checks (saved gold, counts, text hashes);
  - it excludes text-only items by text match.
- **Counts:**
  - 37 files match the pattern, and 7 more are listed explicitly, so 44 rows in all.
  - Of the 44: 29 are test-using, all inside the existing exclusion; 10 use train or validation rows only; 4 are not
    item-level; 1 is recoverable only as text.
- **Newly excluded items:** none.
  - The only text-only file is `clm_template_samples.json`. Its producer script isn't in the repo.
  - Its one Banking77 utterance is mteb row 204, which is position 0 of the seed-0 shuffle and already excluded.
- **Result:** the exclusion sets and the eligible pools are unchanged from clarification 2.
- **Why test ranges can't grow the exclusion:** the excluded spans are positions 0–1999 of the seed-7 CLINC150 shuffle
  and positions 0–1999 of the seed-0 mteb Banking77 shuffle, plus the first 12 unshuffled mteb rows, and the first 100
  mteb MASSIVE en rows. Every test range below lies inside one of them, and the code asserts this.
- **Not item-level:** `bench_turbovec.json`, `retrieval_2000*.json` and `bench_typed.json` match the pattern only
  through ordinary text: the word "massive" in a passage, the SQuAD misspelling "clincal", and a note about a
  concurrent `tune.py` run.

| File | Suite | Classification | Evidence | How it's covered |
|---|---|---|---|---|
| `bench.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0) | bench.py:59-61,80-83 (mteb/banking77 test .shuffle(0), first n); file env.n_per_suite = 500 | Inside the excluded span (asserted); checked: `env/n_per_suite` = 500; `suites/banking77/gold` = re-derived gold [0:500] |
| `bench_harness_check.json` | Banking77 | test: positions 0–7 of mteb/banking77 test, shuffle(seed=0) | bench.py (same env layout) with --n 8: positions 0..7 | Inside the excluded span (asserted); checked: `env/n_per_suite` = 8; `suites/banking77/gold` = re-derived gold [0:8] |
| `bench_smoke.json` | Banking77 | test: positions 0–11 of mteb/banking77 test, unshuffled file order | bench_smoke_v1.py:62-67 (mteb/banking77 test, list(d)[:n], UNSHUFFLED); env.n_per_suite = 12 | Inside the excluded span (asserted); checked: `env/n_per_suite` = 12 |
| `bench_ece.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0) | bench_ece.py:14,77 (test = build_suites(500), bench.json's 500); validation train.shuffle(2)[2000:2300] (:6) | Inside the excluded span (asserted); checked: `bench_ece_probs.npz` `banking77__bodi__test_Y` = re-derived gold [0:500] |
| `bench_experience.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0) | bench_experience.py:45,66 (build_suites(--n 500)); memory = train (tune_experience.py:43) | Inside the excluded span (asserted); checked: `suites/banking77/bodi_experience_per_suite/n` = 500 |
| `bench_experience_gated.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0) | bench_experience.py:45,66 (--gate-file run) | Inside the excluded span (asserted); checked: `suites/banking77/bodi_experience_global/n` = 500 |
| `bench_experience_trust.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0) | bench_experience.py:45,66 (--auto-trust run) | Inside the excluded span (asserted); checked: `suites/banking77/bodi_experience_per_suite/n` = 500 |
| `bench_knn_only.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0) | bench_knn_only.py:30 (build_suites(500)) | Inside the excluded span (asserted); checked: `suites/banking77/global/pred` has 500 items |
| `bench_clm_INVALID_degenerate.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0) | bench_clm.py:82,101 (build_suites(--n 500)); file suites.banking77.n = 500 | Inside the excluded span (asserted); checked: `suites/banking77/n` = 500; `suites/banking77/pred` has 500 items |
| `bench_fresh_experience.json` | Banking77 | test: positions 500–999 of mteb/banking77 test, shuffle(seed=0) | bench_fresh_experience.py:74,79,87 (positions start..start+500, start 500) | Inside the excluded span (asserted); checked: `suites/banking77/gold` = re-derived gold [500:1000] |
| `bench_fresh3_experience.json` | Banking77 | test: positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | bench_fresh_experience.py:23-24,74,87 (--start 1000) | Inside the excluded span (asserted); checked: `suites/banking77/gold` = re-derived gold [1000:1500] |
| `bench_fresh4_head.json` | Banking77 | test: positions 1500–1999 of mteb/banking77 test, shuffle(seed=0) | bench_fresh4_head.py:122,138 (positions 1500..1999); selection rows train.shuffle(2)[2300:2600] (:141-143) | Inside the excluded span (asserted); checked: `suites/banking77/gold` = re-derived gold [1500:2000] |
| `latency_ubuntu_i9-9900X.json` | Banking77 | test: positions 0–549 of mteb/banking77 test, shuffle(seed=0) | bench_latency.py:79-80,87 (build_suites(warmup+calls = 550), timing only) | Inside the excluded span (asserted); checked: `calls` = 500; `warmup` = 50 |
| `laya_full_finetuned.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0); positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | finetune_laya_full.py:139-140 (train_suites(2000,300) = mteb train.shuffle(2); build_suites(1500)), :160-161 (test [:500], fresh3 [1000:1500]) | Inside the excluded span (asserted) |
| `laya_full_finetuned_attempt1_oom.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0); positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | finetune_laya_full.py:139-140 (train_suites(2000,300) = mteb train.shuffle(2); build_suites(1500)), :160-161 (test [:500], fresh3 [1000:1500]); banking77 not_run (counted as used anyway) | Inside the excluded span (asserted) |
| `laya_full_finetuned_attempt2.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0); positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | finetune_laya_full.py:139-140 (train_suites(2000,300) = mteb train.shuffle(2); build_suites(1500)), :160-161 (test [:500], fresh3 [1000:1500]); banking77 not_run (counted as used anyway) | Inside the excluded span (asserted) |
| `laya_full_finetuned_attempt3.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0); positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | finetune_laya_full.py:139-140 (train_suites(2000,300) = mteb train.shuffle(2); build_suites(1500)), :160-161 (test [:500], fresh3 [1000:1500]); banking77 not_run (counted as used anyway) | Inside the excluded span (asserted) |
| `laya_full_finetuned_attempt4_fp16math.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0); positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | finetune_laya_full.py:139-140 (train_suites(2000,300) = mteb train.shuffle(2); build_suites(1500)), :160-161 (test [:500], fresh3 [1000:1500]); banking77 not_run (counted as used anyway) | Inside the excluded span (asserted) |
| `laya_head_finetuned_ubuntu.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0); positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | finetune_laya_head.py:118,126-127 (build_suites(1500), build_suites(500)), :147,155 (fresh3 [1000:1500]); train via tune_experience.py:43; file test_items 'bench.json 0..499', fresh3 'positions 1000..1499' | Inside the excluded span (asserted); checked: `suites/banking77/fresh3/pred` has 500 items |
| `laya_head_finetuned_rerun_savehead.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0); positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | finetune_laya_head.py:118,126-127 (build_suites(1500), build_suites(500)), :147,155 (fresh3 [1000:1500]); train via tune_experience.py:43; file test_items 'bench.json 0..499' | Inside the excluded span (asserted) |
| `laya_head_finetuned_banking77_record_repro.json` | Banking77 | test: positions 0–499 of mteb/banking77 test, shuffle(seed=0); positions 1000–1499 of mteb/banking77 test, shuffle(seed=0) | finetune_laya_head.py:118,126-127 (build_suites(1500), build_suites(500)), :147,155 (fresh3 [1000:1500]); train via tune_experience.py:43; file fresh3 'positions 1000..1499' | Inside the excluded span (asserted) |
| `calibration_train.json` | Banking77 | train/validation rows only | calibrate_train.py:16,23 (learn(calibrate=200) on suites(2000,300) memory = mteb TRAIN .shuffle(2)[:2000]) | Nothing to exclude |
| `check_agree_real.json` | Banking77 | train/validation rows only | check_agree_real.py:14,19-20 (memory and val from suites(2000,300): TRAIN .shuffle(2)[0:2300]) | Nothing to exclude |
| `probe_minilm_knn.json` | Banking77 | train/validation rows only | probe_minilm_knn.py:11-17 (suites(2000,300) mem/val: TRAIN) | Nothing to exclude |
| `tune_knn_only.json` | Banking77 | train/validation rows only | tune_knn_only.py:10 (npz cache); tune_experience.py:33-44,149,156 (suites(2000,300): mteb/banking77 TRAIN .shuffle(2)[0:2300]; npz cache tune_text_*_m2000_v300) | Nothing to exclude |
| `tune_agree_gate.json` | Banking77 | train/validation rows only | tune_agree_gate.py:26 (npz cache); tune_experience.py:33-44,149,156 (suites(2000,300): mteb/banking77 TRAIN .shuffle(2)[0:2300]; npz cache tune_text_*_m2000_v300) | Nothing to exclude |
| `tune_margin_gate.json` | Banking77 | train/validation rows only | tune_margin_gate.py:14 (npz cache); tune_experience.py:33-44,149,156 (suites(2000,300): mteb/banking77 TRAIN .shuffle(2)[0:2300]; npz cache tune_text_*_m2000_v300) | Nothing to exclude |
| `tune_memory_first.json` | Banking77 | train/validation rows only | tune_memory_first.py:30,33 (npz cache + calibration_train.json); tune_experience.py:33-44,149,156 (suites(2000,300): mteb/banking77 TRAIN .shuffle(2)[0:2300]; npz cache tune_text_*_m2000_v300) | Nothing to exclude |
| `tune_experience_text.json` | Banking77 | train/validation rows only | tune_experience.py:33-44,149,156 (suites(2000,300): mteb/banking77 TRAIN .shuffle(2)[0:2300]; npz cache tune_text_*_m2000_v300); test split read for label names only (tune_experience.py:42) | Nothing to exclude |
| `tune_experience_text_trust.json` | Banking77 | train/validation rows only | tune_experience.py:33-44,149,156 (suites(2000,300): mteb/banking77 TRAIN .shuffle(2)[0:2300]; npz cache tune_text_*_m2000_v300) (--auto-trust) | Nothing to exclude |
| `tune.json` | Banking77 | train/validation rows only | tune.py:33-34 (mteb/banking77 TRAIN .shuffle(1)[:300]; test split for label names only); file splits.banking77 = 'train seed=1' | Nothing to exclude |
| `bench_typed.json` | Banking77 | not item-level | bench_typed.py:32 (LocalLLaMA/typed-decisions); 'banking77' only in concurrent_load note | Nothing to exclude |
| `clm_template_samples.json` | Banking77 | items recoverable only as text | producer script not in the repo; samples.banking77.state_text holds one utterance ('message: ...') | Utterance extracted and matched to PolyAI rows by text; it is mteb row 204 = seed-0 position 0, already excluded, so nothing is added |
| `bench_clinc.json` | CLINC150 | test: positions 0–999 of clinc_oos/plus test, shuffle(seed=7) | bench_clinc.py:30-32,87 (test .shuffle(7) first 1000); validation .shuffle(7)[:600] (:79) | Inside the excluded span (asserted); checked: `gold` = re-derived gold [0:1000]; `test_n` = 1000 |
| `bench_clinc_oos.json` | CLINC150 | test: positions 1000–1999 of clinc_oos/plus test, shuffle(seed=7) | bench_clinc_oos.py:76-84 (test .shuffle(7) positions 1000..1999; validation oos + shuffle(7) in-scope) | Inside the excluded span (asserted); checked: `test_n` = 1000; `test/C/pred` has 1000 items |
| `bench_clinc_oos_arrays.json` | CLINC150 | test: positions 1000–1999 of clinc_oos/plus test, shuffle(seed=7) | bench_clinc_oos.py:84,109-111; file test_items 'test.shuffle(7) positions 1000..1999' | Inside the excluded span (asserted); checked: `test/gold` = re-derived gold [1000:2000]; `test_items` = 'test.shuffle(7) positions 1000..1999' |
| `check_oos_embedder.json` | CLINC150 | test: positions 1000–1999 of clinc_oos/plus test, shuffle(seed=7) | check_oos_embedder.py:18-20 (same 1000..1999) | Inside the excluded span (asserted); checked: `test/n` = 1000 |
| `check_oos_product_prefix_build.json` | CLINC150 | test: positions 1000–1999 of clinc_oos/plus test, shuffle(seed=7) | check_oos_product.py:26-28 (same 1000..1999) | Inside the excluded span (asserted); checked: `test/texts_sha256` = SHA of re-derived texts [1000:2000] |
| `check_oos_product_ubuntu.json` | CLINC150 | test: positions 1000–1999 of clinc_oos/plus test, shuffle(seed=7) | check_oos_product.py:26-28 (same 1000..1999) | Inside the excluded span (asserted); checked: `test/texts_sha256` = SHA of re-derived texts [1000:2000] |
| `check_oos_product.json` | CLINC150 | test: positions 1000–1999 of clinc_oos/plus test, shuffle(seed=7) | check_oos_product.py:26-28,82 default output (present only where it was run) | Inside the excluded span (asserted); checked: `test/texts_sha256` = SHA of re-derived texts [1000:2000]. Optional: checked only where the file exists |
| `retrieval_2000.json` | CLINC150 | not item-level | bench_retrieval.py:94 (rajpurkar/squad); regex hit is the misspelling 'clincal' in a SQuAD question | Nothing to exclude |
| `retrieval_2000_qfix.json` | CLINC150 | not item-level | bench_retrieval.py:94 (rajpurkar/squad); regex hit is 'clincal' in a SQuAD question | Nothing to exclude |
| `bench_massive.json` | MASSIVE | test: positions 0–99 of mteb/amazon_massive_intent en test, unshuffled file order | bench_massive.py:41,45 (mteb/amazon_massive_intent <lang> test, list(d)[:per_lang]); file per_lang = 100 | Inside the excluded span (asserted); checked: `per_lang` = 100; `per_language/en/gold` = re-derived gold [0:100] |
| `bench_turbovec.json` | MASSIVE | not item-level | bench_turbovec.py (PREREG_TURBOVEC corpus); regex hit is the word 'massive' in a passage | Nothing to exclude |

**Clarification 3 addendum (2026-10-01, before items are built):** `retrieval_fastmemory_p2000.json` was produced
after the audit (the fastmemory SQuAD retrieval run). It matches the audit pattern only through the SQuAD misspelling
"clincal" in a question, the same as `retrieval_2000.json`, so it is classified "not item-level". The items phase
stopped on it as designed, and resumes after this entry.
