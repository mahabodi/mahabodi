"""Second trigger for experience memory (reviewer suggestion), tuned on VALIDATION only (cached arrays
from tune_experience.py, text space - the same cache tune_margin_gate.py uses).

Base = tune_margin_gate.json best (memory pooled in only when Laya's top1-top2 margin < tau).
Added: memory may override even a CONFIDENT Laya answer when
  (a) at least `agree` of the k=8 nearest stored examples carry the same label, AND
  (b) the task's memory trust >= `trust`, where trust = leave-one-out kNN accuracy of the stored
      examples themselves (each stored example classified by its 8 nearest OTHER stored examples;
      needs no validation labels, so it is available at run time).
Override mode: "pool" (the log-linear pool) or "vote" (the neighbours' majority label).
Constraint: no suite more than 0.5 point below Laya's own validation accuracy; objective = mean.

    .venv/bin/python research/tune_agree_gate.py
"""
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import simulate  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "research", "cache")
R = os.path.join(ROOT, "research", "results")
gb = json.load(open(os.path.join(R, "tune_margin_gate.json")))["best"]
K = 8
S = {}
for name in ["ag_news", "emotion", "banking77", "sst5", "prompt_injections", "boolq"]:
    z = np.load(os.path.join(C, "tune_text_%s_m2000_v300.npz" % name))
    Mv, My, E, P, Y = z["Mv"], z["My"], z["E"], z["P"], z["Y"]
    n = P.shape[1]
    sims = E @ Mv.T
    top = np.argsort(-sims, 1)[:, :K]
    votes = np.stack([np.bincount(My[t], minlength=n) for t in top])
    ms = Mv @ Mv.T
    np.fill_diagonal(ms, -np.inf)
    mt = np.argsort(-ms, 1)[:, :K]
    loo = float(np.mean([np.bincount(My[t], minlength=n).argmax() == y for t, y in zip(mt, My)]))
    srt = np.sort(P, 1)
    margin = srt[:, -1] - srt[:, -2]
    Pm = simulate(P, E, Mv, My, n, gb["k"], gb["temperature"], gb["weight"])
    base_final = np.where((margin < gb["tau"])[:, None], Pm, P)
    S[name] = dict(P=P, Y=Y, Pm=Pm, votes=votes, loo=loo, base_final=base_final)
    print(name, "loo_trust", round(loo, 4), flush=True)
rows = []
for agree, trust, mode in itertools.product([6, 7, 8], [0.6, 0.7, 0.8, 0.9, 1.01], ["pool", "vote"]):
    accs, worst = {}, 0.0
    for name, d in S.items():
        fire = (d["votes"].max(1) >= agree) & (d["loo"] >= trust)
        alt = d["Pm"].argmax(1) if mode == "pool" else d["votes"].argmax(1)
        pred = np.where(fire, alt, d["base_final"].argmax(1))
        acc = float((pred == d["Y"]).mean()); base = float((d["P"].argmax(1) == d["Y"]).mean())
        accs[name] = round(acc, 4); worst = min(worst, acc - base)
    rows.append({"agree": agree, "trust": trust, "mode": mode, "mean": round(float(np.mean(list(accs.values()))), 4),
                 "worst_vs_laya": round(worst, 4), "per_suite": accs})
ok = [r for r in rows if r["worst_vs_laya"] >= -0.005]
best = max(ok, key=lambda r: (r["mean"], r["trust"], r["agree"]))
gated_only = {name: round(float((d["base_final"].argmax(1) == d["Y"]).mean()), 4) for name, d in S.items()}
res = {"base_gate": gb, "loo_trust": {n: round(d["loo"], 4) for n, d in S.items()},
       "laya_val": {n: round(float((d["P"].argmax(1) == d["Y"]).mean()), 4) for n, d in S.items()},
       "gated_only_val": gated_only, "best": best, "grid": rows}
json.dump(res, open(os.path.join(R, "tune_agree_gate.json"), "w"), indent=1)
print("laya", res["laya_val"]); print("gated", gated_only); print("best", best)
