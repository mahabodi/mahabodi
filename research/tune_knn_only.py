"""Tune the memory-alone (kNN) baseline's own k and temperature on the SAME validation data and
memory as tune_experience.py (cached arrays), so bench_knn_only.py compares against a baseline
tuned as carefully as the fusion. Test data is not read."""
import itertools, json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "research", "cache")
out = {}
for name in ["ag_news", "emotion", "banking77", "sst5", "prompt_injections", "boolq"]:
    z = np.load(os.path.join(C, "tune_text_%s_m2000_v300.npz" % name))
    Mv, My, E, Y = z["Mv"], z["My"], z["E"], z["Y"]
    n = max(int(My.max()), int(Y.max())) + 1
    sims = E @ Mv.T
    best = None
    for k, T in itertools.product([1, 4, 8, 16, 32, 64], [0.02, 0.05, 0.1, 0.2, 1.0]):
        top = np.argsort(-sims, 1)[:, :k]
        pred = [np.bincount(My[top[i]], weights=np.exp((sims[i, top[i]] - sims[i, top[i]].max()) / T), minlength=n).argmax() for i in range(len(E))]
        acc = float((np.array(pred) == Y).mean())
        if best is None or acc > best["accuracy"]:
            best = {"k": k, "temperature": T, "accuracy": round(acc, 4)}
    out[name] = best
    print(name, best)
json.dump(out, open(os.path.join(ROOT, "research", "results", "tune_knn_only.json"), "w"), indent=1)
