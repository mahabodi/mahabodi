"""Choose experience_memory_first_margin on VALIDATION (cached arrays from tune_experience.py).

Memory-first rule (decide.rs): for a calibrated question, if the memory's leave-one-out accuracy
(best k, T on the stored cases) minus Laya's accuracy on labelled calibration cases >= margin,
the memory's kNN answers; otherwise the default (margin gate + agreement override) answers.
Selection mirrors the product: Laya's calibration accuracy, the memory's LOO accuracy and its (k, T)
all come from the real calibrate() on the STORED train cases (calibration_train.json, from
calibrate_train.py). Validation items are used only to SCORE each margin, never to select.
Objective: mean validation accuracy over 6 suites, no suite below Laya's validation accuracy by
more than 0.5 point; ties -> larger (more conservative) margin.

    .venv/bin/python research/tune_memory_first.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import simulate  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "research", "cache"); R = os.path.join(ROOT, "research", "results")
gb = json.load(open(os.path.join(R, "tune_margin_gate.json")))["best"]
ab = json.load(open(os.path.join(R, "tune_agree_gate.json")))["best"]
KS, TS = [1, 2, 4, 8, 16, 32], [0.02, 0.05, 0.1, 0.2, 1.0]


def knn_pred(sims, My, n, k, T):
    top = np.argsort(-sims, 1)[:, :k]
    return np.array([np.bincount(My[t], weights=np.exp((sims[i, t] - sims[i, t].max()) / T), minlength=n).argmax() for i, t in enumerate(top)])


CAL = json.load(open(os.path.join(R, "calibration_train.json")))
S = {}
for name in ["ag_news", "emotion", "banking77", "sst5", "prompt_injections", "boolq"]:
    z = np.load(os.path.join(C, "tune_text_%s_m2000_v300.npz" % name))
    Mv, My, E, P, Y = z["Mv"], z["My"], z["E"], z["P"], z["Y"]; n = P.shape[1]
    ms = Mv @ Mv.T; np.fill_diagonal(ms, -np.inf)
    c = CAL[name]
    k, T, la = c["memory_k"], c["memory_temperature"], c["memory_loo_accuracy"]
    sims = E @ Mv.T
    mem_pred = knn_pred(sims, My, n, k, T)
    # default path (gate + agreement override), as tune_agree_gate.py simulates it
    srt = np.sort(P, 1); margin = srt[:, -1] - srt[:, -2]
    Pm = simulate(P, E, Mv, My, n, gb["k"], gb["temperature"], gb["weight"])
    base = np.where((margin < gb["tau"])[:, None], Pm, P).argmax(1)
    top8 = np.argsort(-sims, 1)[:, :8]
    votes = np.stack([np.bincount(My[t], minlength=n) for t in top8])
    loo8 = float(np.mean([np.bincount(My[t], minlength=n).argmax() == y for t, y in zip(np.argsort(-ms, 1)[:, :8], My)]))
    fire = (votes.max(1) >= ab["agree"]) & (loo8 >= ab["trust"])
    default_pred = np.where(fire, votes.argmax(1), base)
    laya_acc = float((P.argmax(1) == Y).mean())
    S[name] = dict(laya=c["laya_accuracy"], laya_val=laya_acc, loo=la, k=k, T=T, mem=float((mem_pred == Y).mean()), default=float((default_pred == Y).mean()))
    print(name, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in S[name].items()}, flush=True)
rows = []
for m in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 1.01]:
    acc = {n_: (d["mem"] if d["loo"] - d["laya"] >= m else d["default"]) for n_, d in S.items()}
    worst = min(acc[n_] - S[n_]["laya_val"] for n_ in S)
    rows.append({"margin": m, "mean": round(float(np.mean(list(acc.values()))), 4), "worst_vs_laya": round(worst, 4),
                 "memory_first_suites": [n_ for n_, d in S.items() if d["loo"] - d["laya"] >= m], "per_suite": {k: round(v, 4) for k, v in acc.items()}})
    print(rows[-1], flush=True)
ok = [r for r in rows if r["worst_vs_laya"] >= -0.005]
best = max(ok, key=lambda r: (r["mean"], r["margin"]))
json.dump({"suites": S, "grid": rows, "best": best}, open(os.path.join(R, "tune_memory_first.json"), "w"), indent=1)
print("best", best)
