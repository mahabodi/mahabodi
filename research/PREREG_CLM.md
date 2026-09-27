# Pre-registration: CLM-8B vs Laya vs MahaBodi on our hardware

Written and agreed with the reviewer **before any CLM prediction is scored**. The user asked to compare CLM
([Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) at bb42c6c; model
[CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)) with Laya and MahaBodi **on our own resources**.

## What CLM claims (third-party, not measured here)

CLM's README compares CLM-8B with **Jev**, not with Laya. Its tool-calling result (BFCL v4) is 76.8 ms vs Jev's
125.5 ms, with success 95.2 % vs Jev's 99.2 %, on an RTX 4090 / H100. The repo has no Laya comparison and no
"13×" figure. Those numbers are cited as third-party claims only.

## Hardware (the user's resources)

- Ubuntu box: i9-9900X (20 logical CPUs), 61 GB RAM, RTX 2080 Ti **11 GB** (capped at 200 W).
- CLM's encoder is **Qwen3-8B**, about 16 GB of weights in 16-bit, which does not fit on the 11 GB GPU. It runs on
  the **CPU in fp32** (transformers, about 32 GB of RAM): unquantised, so its embeddings are CLM's own.
- **The headline comparison is CPU-only for every system on the same machine** (Laya PyTorch, Laya-identical ONNX,
  MahaBodi, CLM), with the same thread count.
- A GPU number for CLM on this box is not possible without quantisation, which would change the model. It is
  therefore **not reported**, and CLM's published GPU latencies are not compared with our CPU ones.

## Faithful encoder (parity check before scoring)

- The vLLM pooling path (`--runner pooling`, Qwen3-8B, last-token pooling, then L2-normalise in `clm/embedder.py`)
  is reproduced with transformers: final-layer hidden state (after the model's final norm) at the last token, no
  special tokens added, truncated to 2,048 tokens, L2-normalised.
- **Parity, before any scoring:**
  - (a) **End to end.** Reproduce the example outputs documented in CLM's README (`urgency` noul 0.41022, `department`
    billing 0.93878, `frustration` score 1.98386) through our encoder + `clm.Engine`, within 0.005 each. This checks
    encoder, head and question schema together.
  - (b) **Row level.** Where published CLM embeddings come with their source text, ≥100 rows must be re-embedded
    at min cosine ≥ 0.999. The public `Contrastive-LM` embedding sets checked so far are DeepSWE (no text) and
    Nemotron pretrain (only doc ids into a separate Nemotron Q&A set). If no text can be matched, (b) is reported
    "not possible" and (a) stands alone.
  - If (a) fails, CLM's accuracy is reported as "unverified encoder parity".

## Systems and items

- **CLM-8B:** `clm.Engine.answer(state, questions)` with the reference head `clm-latest` (CLM_v0.1-8B.pt),
  zero-shot, temperature 1; our encoder is behind a local `/v1/embeddings` endpoint.
- **Laya:** its saved per-item predictions (PyTorch, the published checkpoints), re-scored on the same items.
- **MahaBodi:** its saved per-item zero-shot `decide()` predictions (defaults), on the same items.
- **Suites** (the exact seeded test items already used in BENCHMARKS.md): ag_news, emotion, banking77, sst5 and
  boolq (500 each); prompt_injections (116); typed-decisions (`bench_typed.json`); MASSIVE (`bench_massive.json`,
  if time allows: 5,100 items).
- **Questions are posed identically** to how Laya is asked in `research/bench.py` (the same state text, instructions
  and option criteria), converted to CLM's wire format by CLM's own client types.

## Pinning, licence and determinism

- **Pinned:** CLM repo commit bb42c6c; the Hugging Face revision hashes of `Qwen/Qwen3-8B` and
  `Contrastive-LM/CLM-v0.1-8B` (`clm-latest` is recorded as that concrete revision); sha256 of every weight file;
  transformers and torch versions.
- **Licences:** the CLM code is Apache-2.0 (LICENSE), the CLM-8B weights Apache-2.0 (model card), and Qwen3-8B
  Apache-2.0. Recorded in the result file. Benchmarking and publishing results is allowed.
- **Temperature:** CLM's `temperature` (default 1) divides the logits before a softmax. It is a calibration
  parameter, not sampling, so CLM's output is deterministic; this is verified by running 50 items twice and
  requiring identical outputs. Per item, CLM's label and full probability vector are recorded.

## Metrics

- Accuracy with Wilson 95 % CIs.
- Exact McNemar per suite: CLM vs Laya, CLM vs MahaBodi.
- **Latency is measured FRESH for all three systems** on the same idle Ubuntu CPU (no GPU or CPU job active, checked
  in the health log), with the same threads and the same items. The saved predictions carry no latency and came from
  another machine.
- **Accuracy for Laya and MahaBodi** may come from the saved per-item predictions only if a seeded 100-item subset
  per suite, re-run on Ubuntu, gives identical predictions; otherwise those suites are re-run in full.
- Per-decision latency p50/p95 on the same CPU with 8 threads, cache off, reported separately for:
  - a cold state (encoder call);
  - cached options (CLM's design caches option embeddings, and that is disclosed as its advantage).
- Peak RSS and load time. **Model sizes are stated next to latency** (CLM: Qwen3-8B fp32 + 20M head; Laya's
  checkpoint size; MahaBodi = Laya + MiniLM), so a speed gap is not read as a method difference alone.

## Rules

- Zero-shot only: no fine-tuning of CLM, Laya or MahaBodi for this comparison.
- **Contamination is disclosed:**
  - Laya was trained on the training splits of its own suites (ag_news/boolq validation also; see
    BENCHMARKS.md).
  - CLM was trained on Nemotron Q&A plus agentic traces, and its repo ships a fine-tuning recipe for
    LocalLLaMA/typed-decisions (not used here).
- **Tool calling (BFCL v4)** is run only if a reproducible harness for all three systems can be written on the same
  items. Otherwise it is "not run", with the reason. CLM's 95.2 % is never compared with our numbers from a
  different harness.
- **Wording:** the "13× faster than Laya on tool calling" claim appears nowhere in CLM's materials. It is never
  attributed to CLM; if mentioned at all, it is "a claim we could not find a source for".
- **Per-suite verdicts** (p < 0.05), with no pooled headline. CLM's wins and losses are reported alike.
- The reviewer recomputes everything from per-item files before any wording.
