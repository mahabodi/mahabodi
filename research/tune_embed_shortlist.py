"""Zero-shot embedding shortlist for banking77, tuned on VALIDATION only (no experience memory,
no labelled data used by the decision): val = banking77 train[seed2][2000:2300] (same slice
as the other tuning scripts). Configs replace the tournament with the top-N options by MiniLM
similarity to the message, or add them to the tournament finalists (union).

    .venv/bin/python research/tune_embed_shortlist.py
"""
import json, os, sys, time
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import suites  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "results", "tune_embed_shortlist.json")
S = suites(2000, 300)["banking77"]
qid = next(iter(S["q"]))
b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
val = [S["st"](r) for r in S["val"]]; Y = [S["y"](r) for r in S["val"]]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for n, union in [(16, False), (12, False), (8, False), (8, True), (16, True)]:
    key = "n%d_%s" % (n, "union" if union else "replace")
    if key in res:
        continue
    s0 = b.stats()["system1"]["rows_scored"]; t0 = time.time()
    out = b.decide_batch(val, S["q"], cache=False, experience_k=0, embed_shortlist=n, embed_shortlist_union=union)
    acc = sum(o["answers"][qid]["choice"] == y for o, y in zip(out, Y)) / len(Y)
    res[key] = {"accuracy": round(acc, 4), "rows_per_decision": round((b.stats()["system1"]["rows_scored"] - s0) / len(Y), 2), "seconds": round(time.time() - t0, 1)}
    print(key, res[key], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
