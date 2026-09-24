<p align="center">
  <img src="assets/logo/mahabodi-banner.svg" alt="MahaBodi, an AI Bodhi tree: fastmemory graph memory as the roots, the MahaBodi engine as the trunk, six language bindings as the branches, Laya decisions as the leaves" width="100%">
</p>

# MahaBodi

**A System-1 engine for AI agents.** MahaBodi gives an agent two fast, non-generative
faculties behind one API:

* **Memory** – [fastmemory](https://github.com/fastBuilderAI/memory)'s topology memory
  (ATFs clustered into blocks by Louvain), with a **concept-density guard** and a **query
  cascade** that reports how, and whether, every query matched.
* **Decisions** – [Laya](https://github.com/NandhaKishorM/laya)'s calibrated `choice` /
  `score` / `noul` decision model, executed natively in Rust through ONNX Runtime, with
  tournament shortlisting for many-option questions, a non-Latin script guard, and an
  answer cache.

The core is Rust (`crates/mahabodi-core`), with bindings for **Python, Node.js, Java, C#/.NET
and Go** plus a C ABI.

> Status: pre-release, built from source only. Nothing is published to PyPI, npm, crates.io,
> Maven Central, NuGet or the Go proxy yet. crates.io publishing is blocked until fastmemory
> releases `cluster::run_louvain_inline` (MahaBodi currently pins fastmemory to git rev
> `a7dec441`).

## Benchmark highlights

Every figure below is from [BENCHMARKS.md](BENCHMARKS.md): same machine, same Laya checkpoint on
both sides, seeded test samples (n = 500 per suite unless noted), and an exact McNemar test on
the same items. A result counts as a **beat** only at p < 0.05.

### At a glance

| Result | Laya (or baseline) | MahaBodi | Verdict (exact McNemar) |
|---|---|---|---|
| MASSIVE intent, 51 languages, zero-shot | 0.366 macro, 45/51 languages usable | **0.405 macro, 47/51 usable** | **beat**, p < 1e-6 |
| Banking77 (77 intents), zero-shot | 0.492 | **0.660** | **beat**, p < 1e-6 (tie vs Laya + MiniLM shortlist) |
| Banking77 with 2,000 labelled examples | 0.492 (zero-shot) | **0.888** | **beat**, different setting (a plain kNN over the same examples scores 0.872: tie) |
| Misspelled keyword retrieval (SQuAD, 300 paragraphs) | baseline BM25: 0.05 recall@5 (Laya does no retrieval) | **0.61** | **beat** |
| BoolQ answered from memory (question only in, passage retrieved) | 0.424 (question only); always-yes 0.626 | **0.782** | **beat** both, p < 1e-6; below the oracle passage (0.846) |
| 6 other Laya suites, zero-shot | = | = | tie: exact parity |

Zero-shot scorecard against Laya's 10 published benchmarks: **2 beats, 6 ties, 2 not yet run**
(idle-machine latency; calibration as a scored row). MahaBodi runs Laya's own models: the wins come
from how MahaBodi uses them (tournament shortlisting, experience memory, retrieval), not from a
new model.

**1. Multilingual, zero-shot: MahaBodi beats Laya on Laya's own 51-language benchmark.** On
MASSIVE intent (20 options, 100 utterances in each of 51 languages, Laya's own construction),
with the `laya-multilingual` checkpoint on both sides:

| | Laya | MahaBodi |
|---|---|---|
| macro accuracy | 0.366 | **0.405** |
| languages above 3× random | 45 / 51 | **47 / 51** |

The difference is significant: exact McNemar over all 5,100 items, 541 vs 344 discordant,
p < 1e-6. The Laya column reproduces Laya's published figures exactly (0.366 and 45/51). The gain
comes from tournament shortlisting, whose settings were fixed on English Banking77 and not tuned
on MASSIVE. Its cost is 4 model passes per decision, against Laya's 1. [Details](BENCHMARKS.md#massive-intent-20-options-per-language-laya-multilingual-checkpoint-on-both-sides)

**2. Many-option decisions, zero-shot: +16.8 points over Laya on Banking77.** On 77 intents,
MahaBodi's tournament shortlisting scores **0.660 against Laya's 0.492** (p < 1e-6). It also
beats two of Laya's own documented mitigations: a 512-token head budget (0.564) and shortlisting
with Laya's encoder (0.268). It **ties** Laya's shortlist with a MiniLM embedder (0.642, p = 0.30),
which is also faster. [Details](BENCHMARKS.md#banking77-mahabodi-vs-layas-own-many-option-mitigations)

**3. Exact Laya parity, natively in Rust.** On the other six published suites MahaBodi
reproduces Laya's decisions item for item: typed-decisions 0.766, AG News 0.934, Emotion 0.604,
SST-5 0.350, prompt-injections 0.698, BoolQ 0.846, with zero disagreements. These are ties by
construction, not wins. [Details](BENCHMARKS.md)

**4. Experience memory: learning from labelled examples without retraining.** Given up to 2,000
labelled examples per task (a different setting from zero-shot), the per-task setting beats
Laya zero-shot on four of six suites:

| Suite | Laya (zero-shot) | MahaBodi + experience memory |
|---|---|---|
| Banking77 | 0.492 | **0.888** |
| Emotion | 0.604 | **0.668** |
| SST-5 | 0.350 | **0.430** |
| prompt-injections | 0.698 | **0.767** |
| AG News, BoolQ | 0.934, 0.846 | tie |

A tuned kNN over the same examples does about as well on several of these suites. The
**default** setting is margin-gated: memory changes only answers Laya is unsure of. It has **no
loss against Laya on any of the six suites** (beats it on Banking77, Emotion and SST-5; ties on
the rest). It still loses to the kNN memory alone on Banking77 and prompt-injections.
[Details](BENCHMARKS.md#with-experience-memory-a-different-setting-uses-labelled-examples)

**5. Breaking Laya's near-ties.** When Laya's top two options are within 0.10 of each other it
is only 17–42 % accurate. Labelled memory fixes most of these: Banking77 near-ties go from
**26 % to 91 %** and SST-5 from 18 % to 33 %, both significant.
[Details](Enterprise.md#6-accuracy-hallucination-and-near-ties-what-grounding-does-and-does-not-guarantee)

**6. Typo-tolerant memory retrieval.** On held-out SQuAD questions, hybrid retrieval finds the
right passage in the top 5 for **61 % of misspelled keyword queries, against 5 % for BM25**
(300 paragraphs; 40 % against 1.5 % at 2,000). On clean questions it beats BM25 at 300
paragraphs (0.973 vs 0.952) and ties it at 2,000. On plain keyword queries at 2,000 paragraphs it
**loses** to BM25. Confidently wrong answers remain on hard queries.
[Details](BENCHMARKS.md#memory-retrieval-held-out-squad-validation-questions-a-hub-fallback-counts-as-a-miss)

**7. Decisions grounded in memory.** Grounding decisions in MahaBodi memory lifts Laya on BoolQ
from **0.424** (question only) to **0.782**, significantly above always answering yes (0.626).
Confidently wrong answers fall from 45 % to 17 %. It does not reach the oracle passage (0.846).
Memory held each question's own passage among 500, so this is retrieval over a relevant knowledge
base, not open-domain QA. The passage format was chosen on a separate dev sample.
[Details](BENCHMARKS.md#grounded-decisions-boolq-answered-from-mahabodi-memory-english-laya-checkpoint)

Not yet claimed: latency on an idle machine; calibration (ECE) as a scored row.
Deploying at TB/PB scale on PostgreSQL + Apache AGE with separately hosted models is covered in
[Enterprise.md](Enterprise.md).

## What it does

```text
text / ATF markdown / fastmemory entity tags
        │ ingest (per-region format detection; never drops content)
        ▼
fastmemory ATFs ──► edges (fastmemory's rule + context links + density concepts)
        │                 │
        │                 └─► Louvain communities (deterministic port of fastmemory's inline Louvain)
        ▼
concept-density guard ──► BM25 / stem / trigram index
        │
        ▼
query cascade: exact → substring (fastmemory rule) → stem → fuzzy → hub
   every result says matched / handoff / stage / confidence
        │
        ▼  context (empty on handoff — never grounds a decision on unrelated memory)
Laya decision (ONNX, Rust) ──► typed answer + probabilities + confidence (+ handoff flags)
```

### Memory: density guard and "queries never fail", made precise

fastmemory's Rust parser only reads entity tags such as `(Function Validate_Token)`. Its own
README format (`## [ID: x]` + `**Action:**` fields) and plain prose parse to *zero* ATFs, and
ATFs without data/access/event links are dropped by its Louvain step. Either way, queries on
that memory find nothing. MahaBodi:

* ingests all three input shapes, deciding per region, and falls back to prose so content is
  never lost;
* keeps every ATF as a node, even when it has no edges;
* measures concept density (isolated ATFs, links per ATF, and a self-probe that asks whether
  each ATF is found by its own rarest term) and repairs it by linking under-connected ATFs to
  shared concept nodes;
* answers every query on a non-empty memory, but **never pretends**: each result carries
  `matched`, `handoff` (hand this to System 2) and `stage`. A `hub` fallback, a query with no
  content terms, or one that covers under half of its terms is flagged `handoff: true`, and
  `context()` returns nothing for it.

"Never fails" therefore means *never errors and never silently returns an unrelated answer as
a match*. Retrieval quality is measured separately on held-out questions
(`research/bench_retrieval.py`).

### Decisions: Laya, natively

`predict()` reproduces Laya's English checkpoint exactly: token ids are identical and every
probability is within 2e-4 of Laya's PyTorch model on the golden parity cases
(`cargo test -p mahabodi-core --release --test laya_parity -- --ignored`). `decide()` adds:

* **Tournament shortlisting.** Laya gives all options of a question one shared token budget,
  so at 77 labels each label keeps about 4 tokens. MahaBodi scores the options in groups,
  keeps the top options of each group, and decides among the finalists. It costs more model
  rows per decision; the benchmarks report both accuracy and rows.
* **A non-Latin script guard.** The English checkpoint is never run on text it cannot read.
  Such answers come back with a `null` decision and `handoff: true`. This is a *subset* of
  Laya's Router: Latin-script non-English text is not detected.
* **An answer cache,** plus an optional adaptive option-order ensemble. The ensemble is off by
  default because it showed no gain on validation data.

## Build and test

Requirements: Rust 1.88+, libonnxruntime 1.23+ (for decisions), Python 3.11 venv (for model
export and benchmarks). Optional per binding: maturin, Node 18+, JDK 11+ and Maven, .NET 8 SDK, Go 1.21+.

```bash
python3.11 -m venv .venv && .venv/bin/pip install torch==2.2.2 "numpy<2" transformers==4.57.3 \
    safetensors huggingface_hub onnx onnxruntime==1.23.2 laya
.venv/bin/python research/export_onnx.py --out models/laya-v2       # ~1.7 GB, verified vs PyTorch
./scripts/test_all.sh                                               # every language, real model
```

`scripts/test_all.sh` prints one PASS/FAIL line per suite: rust, laya_parity, python, node,
java, csharp and go.

## Usage

**Python** (`bindings/python`, `maturin develop --release`)

```python
from mahabodi import Bodi
b = Bodi()
b.ingest(open("kb.md").read(), source="kb")
r = b.query("refund policy")                 # r["matched"], r["handoff"], r["stage"], r["hits"]
b.load_laya("models/laya-v2")
b.decide("I was charged twice", {"refund": {"type": "noul", "instructions": "Asks for a refund?"}})
b.decide_with_memory("I was charged twice", {...}, query="duplicate charge refund")
```

**Node.js** (`bindings/node`, `npm run build`): `const { Bodi } = require('mahabodi')`. Model
calls return Promises and run off the event loop.

**Java** (`bindings/java`, `mvn test`): `try (Bodi b = new Bodi()) { b.query("refunds", 5); }`,
with JSON strings in and out.

**C#** (`bindings/csharp`, `dotnet test`): `using MahaBodi; using var b = new Bodi(); b.Query("refunds");`

**Go** (`bindings/go`): `e, _ := mahabodi.New(nil); r, _ := e.Query("refunds", 5)`. cgo links
`target/release/libmahabodi`.

**C**: `crates/mahabodi-ffi/include/mahabodi.h`, four functions, JSON in and out.

## Benchmarks

Every number in `BENCHMARKS.md` is generated from `research/results/*.json` by
`research/report.py`, and each JSON is produced by one committed script. The runs are on a
single machine (Intel i9-9980HK, x86_64 macOS, CPU only), the samples are seeded, and tuning
uses validation/train splits only. Ties and losses are reported as such.
Laya's published latency (32.8 ms) comes from a T4 GPU and is not comparable to CPU runs here.

## Privacy note

fastmemory's CLI and Python package post a license check, including the hostname and IP
address (via api.ipify.org), to fastbuilder.ai. MahaBodi calls only fastmemory's `parser` and
`cluster` code paths, which do not trigger it. Research scripts that run the fastmemory Python
package do so under `sandbox-exec` with outbound network denied.

## License

[MIT](LICENSE). MahaBodi runs third-party models and code under their own licenses: Laya's models
and MiniLM are Apache-2.0, fastmemory is MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Test fixtures include fastmemory example inputs (MIT; see
`crates/mahabodi-core/tests/fixtures/FASTMEMORY_LICENSE`).
