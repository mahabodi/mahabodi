# MahaBodi (Python)

MahaBodi is a System-1 engine for AI agents, written in Rust. It combines a fastmemory-style topology memory
(ingest, a density guard, a query cascade that always returns an answer or a handoff, and hybrid retrieval)
with fast typed decisions from [Laya](https://github.com/NandhaKishorM/laya) models, run through ONNX Runtime.

```bash
pip install mahabodi            # memory only
pip install "mahabodi[laya]"    # also installs onnxruntime, for Laya decisions
```

```python
from mahabodi import Bodi

b = Bodi()
b.ingest(open("kb.md").read(), source="kb")
r = b.query("refund policy")        # r["matched"], r["handoff"], r["stage"], r["hits"]

# Decisions need a Laya model exported to ONNX (see "Models" below)
b.load_laya("models/laya-v2")
b.decide("I was charged twice", {"refund": {"type": "noul", "instructions": "Asks for a refund?"}})

# Learn from labelled cases without retraining (needs the MiniLM embedder export)
b.load_embedder("models/minilm")
b.learn(states, questions, labels)
```

Every method returns plain dicts and lists; errors raise `ValueError`.

## Models

MahaBodi does not ship model weights. To export Laya's checkpoint and the MiniLM embedder to ONNX, use
`research/export_onnx.py` and `research/export_embedder.py` in the
[repository](https://github.com/mahabodi/mahabodi). ONNX Runtime is found through `ORT_DYLIB_PATH`, or
automatically from the `onnxruntime` package installed in the same Python.

## Results

Every claim is measured against Laya on the same machine for each comparison, with seeded samples and exact McNemar tests, and
losses are reported as losses. On zero-shot Laya benchmarks MahaBodi wins on many-option and multilingual
intent and ties elsewhere (same maths). It loses to a Laya head fine-tuned on SST-5 and to a fully fine-tuned
Laya on emotion. See
[BENCHMARKS.md](https://github.com/mahabodi/mahabodi/blob/main/BENCHMARKS.md) for all results and caveats.

Status: alpha (0.x). MIT license. Third-party notices are in the repository.
