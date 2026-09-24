"""W10.1 margin gate for experience memory, tuned on VALIDATION only (cached arrays from
tune_experience.py, text space). Memory is applied only when Laya's top-1 minus top-2 probability
is below tau; otherwise Laya's answer stands. Grid: tau x (k, T, w) shared across suites; objective
= mean validation accuracy, subject to no suite falling below Laya's own validation accuracy by
more than 0.5 point (the "never worse" principle)."""
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import simulate  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "research", "cache")
S = {}
for name in ["ag_news", "emotion", "banking77", "sst5", "prompt_injections", "boolq"]:
    z = np.load(os.path.join(C, "tune_text_%s_m2000_v300.npz" % name))
    n = z["P"].shape[1]
    S[name] = (z["Mv"], z["My"], z["E"], z["P"], z["Y"], n)
rows = []
for tau, k, T, w in itertools.product([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01], [4, 8, 16], [0.02, 0.05, 0.2], [0.5, 1.0, 2.0, 4.0]):
    accs, worst = {}, 0.0
    for name, (Mv, My, E, P, Y, n) in S.items():
        srt = np.sort(P, 1)
        margin = srt[:, -1] - srt[:, -2]
        Pm = simulate(P, E, Mv, My, n, k, T, w)
        final = np.where((margin < tau)[:, None], Pm, P)
        acc = float((final.argmax(1) == Y).mean()); base = float((P.argmax(1) == Y).mean())
        accs[name] = round(acc, 4); worst = min(worst, acc - base)
    rows.append({"tau": tau, "k": k, "temperature": T, "weight": w, "mean": round(float(np.mean(list(accs.values()))), 4), "worst_vs_laya": round(worst, 4), "per_suite": accs})
ok = [r for r in rows if r["worst_vs_laya"] >= -0.005]
best = max(ok, key=lambda r: (r["mean"], -r["tau"]))
base = {name: round(float((P.argmax(1) == Y).mean()), 4) for name, (Mv, My, E, P, Y, n) in S.items()}
json.dump({"best": best, "laya_val": base, "grid": rows}, open(os.path.join(ROOT, "research", "results", "tune_margin_gate.json"), "w"), indent=1)
print("laya val", base); print("best", best)
