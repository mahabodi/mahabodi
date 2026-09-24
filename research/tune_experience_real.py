"""Banking77 experience configs through the REAL decide path (tournament + memory-proposed
final candidates, which tune_experience.py cannot simulate). Validation only: memory =
banking77 train[seed2][:2000], val = train[seed2][2000:2300] (same split as tune_experience.py).

    .venv/bin/python research/tune_experience_real.py
"""
import json, os, sys, time
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import suites, label_idx, plain_text  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "results", "tune_experience_real.json")
S = suites(2000, 300)["banking77"]
qid, qd = next(iter(S["q"].items()))
b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
mem = [S["st"](r) for r in S["mem"]]
b.learn(mem, S["q"], [{qid: S["y"](r)} for r in S["mem"]])
val = [S["st"](r) for r in S["val"]]
Y = [S["y"](r) for r in S["val"]]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for k, T, w, c in [(16, 0.02, 8.0, 5), (16, 0.02, 4.0, 3), (8, 0.2, 2.0, 3), (16, 0.05, 8.0, 5), (16, 0.02, 16.0, 5)]:
    key = "k%d_T%g_w%g_c%d" % (k, T, w, c)
    if key in res:
        continue
    t0 = time.time()
    out = b.decide_batch(val, S["q"], cache=False, experience_k=k, experience_temperature=T, experience_weight=w, experience_candidates=c)
    acc = sum(o["answers"][qid]["choice"] == y for o, y in zip(out, Y)) / len(Y)
    res[key] = {"accuracy": round(acc, 4), "seconds": round(time.time() - t0, 1)}
    print(key, res[key], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
