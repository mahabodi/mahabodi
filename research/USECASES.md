# New wins: real problems where MahaBodi should beat Laya

These are not Laya's published benchmarks. They are industry problems that exercise what MahaBodi
adds on top of Laya's models.

**MahaBodi's additions:**
- **many options:** the tournament;
- **instant learning from labelled cases:** experience memory;
- **answering from documents:** grounded, hybrid retrieval;
- **knowing when it doesn't know:** handoff.

**Protocol, fixed before any run, for every problem:**
- **Three arms:**
  - (A) Laya alone;
  - (B) Laya plus a standard retriever or shortlist: BM25 or MiniLM top-k, whichever is stronger on validation. This is the fair baseline;
  - (C) MahaBodi.
- **Tuning:** settings are tuned on train/validation only, followed by one seeded test run.
- **Statistics:** exact McNemar (or bootstrap CIs for non-binary metrics).
- **Reporting:** results go in their own BENCHMARKS.md section, verified by the reviewer before any claim.
- **Win condition:** a win means C significantly beats **B**. Beating A alone is reported, but it isn't the claim.

| # | Problem | Benchmark | Where Laya struggles | MahaBodi's edge | Status |
|---|---|---|---|---|---|
| 1 | Agent tool routing across hundreds of tools | ToolBench / API-Bank / Gorilla APIBench | too many options, no tool knowledge | tournament plus tool-doc memory | planned |
| 2 | Intent routing with out-of-scope | CLINC150 (150 intents plus OOS) | 150 options at once; OOS only through a confidence threshold | tournament plus handoff | done (verified; `bench_clinc.json`): tournament beats Laya + MiniLM shortlist on overall accuracy, 0.736 vs 0.708 (p = 0.027, one run, 1,000 items); **handoff failed**: out-of-scope recall 3 %, worst of the three arms |
| 3 | New labels, learning from a few examples | CLINC150 / Banking77, adding labels incrementally | handles new labels zero-shot from their names/descriptions only | `learn()` adds accuracy from n labelled examples, instantly | related result: with 2,000 examples per task the default never falls below Laya on 5 fresh suites (verified; `bench_fresh_experience.json`), but plain kNN still beats it on Banking77 (0.892 vs 0.832); the incremental n = 0/1/5/10 protocol is not run yet |
| 4 | Contract and policy compliance | ContractNLI, LEDGAR | long documents, 100 options | clause retrieval plus tournament | planned |
| 5 | Fact checking against a knowledge base | FEVER, SciFact, Climate-FEVER | no facts | grounded retrieval | pilot done: BoolQ from memory 0.782 vs question-only 0.424 and always-yes 0.626; below oracle 0.846 (verified; `bench_grounding_v2.json`) |
| 6 | Multi-hop decisions | StrategyQA, HotpotQA yes/no | no retrieval | graph traversal (fastmemory) | planned |
| 7 | Duplicate or known-issue detection | Quora duplicate questions, StackExchange duplicates | no store of past issues | memory retrieval plus noul decisions | planned |

## Pre-registered metrics

- **CLINC150 (#2):**
  - in-scope accuracy;
  - OOS recall;
  - OOS precision;
  - overall accuracy with OOS as a class (the dataset's standard metric).
  - Arm A is Laya plus a **validation-tuned confidence threshold** for OOS (Laya's README recommends confidence gating); it is not Laya without any OOS mechanism.
  - MahaBodi answers OOS by handing off, with its threshold also tuned on the CLINC validation split.
- **Incremental labels (#3):**
  - accuracy on newly added intents at n = 0 (zero-shot, both systems: a new label is just a new option, and Laya needs no retraining), 1, 5 and 10 labelled examples each;
  - accuracy retained on the original intents.
  - Arm A is Laya zero-shot with the new labels as options.
  - Arm B (the fair baseline, and the one the win is judged against) is own-tuned kNN on MiniLM
    embeddings of the same n examples, and also max(kNN, Laya) chosen per task on validation. At
    2,000 examples plain kNN already beats MahaBodi on Banking77, so beating Laya alone is not enough.
- **Tool routing (#1):** top-1 correct tool, handoff rate, confidently-wrong rate.

## Pre-registered: out-of-scope gate (W11), fixed 2026-09-25 before any run

- **Gate:** answer `oos` when the maximum cosine similarity between MiniLM(utterance) and MiniLM of
  each intent name is below `s`. It is zero-shot: no CLINC training utterances are used. The primary
  arm uses intent names only. A variant using the kNN similarity to train utterances is a separate
  setting, reported separately.
- **Fairness:** arms A, B and C each get the same gate, each with its own thresholds.
- **Tuning:** `tau` (top-1 probability) and `s` are tuned jointly on a small 2-D grid. Tuning uses
  all 100 CLINC "plus" validation OOS items plus seeded in-scope validation items. The objective
  is overall 151-class accuracy, reweighted to the test split's documented OOS prevalence of 18.2 %
  (1,000 of 5,500), because validation is only 3.2 % OOS.
- **Test:** fresh test positions 1000..1999 of the seed-7 test shuffle, asserted disjoint from
  0..999.
  - Each arm's decisions run once; per item we save the top-1 label, top-1 probability and
    max-similarity.
  - Thresholds are then applied offline.
- **Win condition:** C+gate beats B+gate on overall accuracy (exact McNemar).
- **Also reported for all arms:**
  - in-scope accuracy;
  - OOS recall and precision;
  - OOS AUROC of the similarity score and of the top-1 probability.
- **If B+gate matches C+gate on OOS,** the gate is not a MahaBodi advantage, and the write-up says so.
