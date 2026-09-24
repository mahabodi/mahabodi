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
| 2 | Intent routing with out-of-scope | CLINC150 (150 intents plus OOS) | 150 options at once; OOS only through a confidence threshold | tournament plus handoff | **next** |
| 3 | New labels, learning from a few examples | CLINC150 / Banking77, adding labels incrementally | handles new labels zero-shot from their names/descriptions only | `learn()` adds accuracy from n labelled examples, instantly | **next** |
| 4 | Contract and policy compliance | ContractNLI, LEDGAR | long documents, 100 options | clause retrieval plus tournament | planned |
| 5 | Fact checking against a knowledge base | FEVER, SciFact, Climate-FEVER | no facts | grounded retrieval | pilot: BoolQ grounding (`bench_grounding.py`) |
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
- **Tool routing (#1):** top-1 correct tool, handoff rate, confidently-wrong rate.
