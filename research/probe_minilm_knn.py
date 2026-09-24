"""Probe: neighbour quality of all-MiniLM-L6-v2 sentence embeddings vs MahaBodi's pooled vector,
measured as kNN-only accuracy on the SAME validation splits as tune_experience.py (k=16, T=0.05).
Pooled-vector numbers come from research/results/tune_experience.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import suites, label_idx
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
out = {}
S = suites(2000, 300)
for name in ["ag_news", "emotion", "banking77", "sst5"]:
    s = S[name]; qd = next(iter(s["q"].values()))
    text = lambda r: " ".join(str(v) for v in s["st"](r).values())
    Mv = m.encode([text(r) for r in s["mem"]], normalize_embeddings=True, batch_size=128)
    E = m.encode([text(r) for r in s["val"]], normalize_embeddings=True, batch_size=128)
    My = np.array([label_idx(qd, s["y"](r)) for r in s["mem"]]); Y = np.array([label_idx(qd, s["y"](r)) for r in s["val"]])
    n = int(My.max()) + 1 if qd["type"] != "noul" else 2
    n = max(n, len(qd.get("criteria", [])) if qd["type"] != "noul" else 2)
    sims = E @ Mv.T; top = np.argsort(-sims, 1)[:, :16]
    pred = []
    for i in range(len(E)):
        w = np.exp((sims[i, top[i]] - sims[i, top[i]].max()) / 0.05)
        pred.append(np.bincount(My[top[i]], weights=w, minlength=n).argmax())
    out[name] = round(float((np.array(pred) == Y).mean()), 4)
    print(name, "minilm knn-only", out[name], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "probe_minilm_knn.json"), "w"), indent=1)
