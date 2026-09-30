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
