"""Tune hybrid retrieval's dense_min_similarity on SQuAD TRAIN (disjoint from the validation
split bench_retrieval.py reports). Objective: answered_correctly_without_handoff minus
confidently_wrong_rate on clean questions and on two misspelled-keyword queries.

    .venv/bin/python research/tune_hybrid.py
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_retrieval import typo_all, keywords  # noqa: E402
from datasets import load_dataset
from mahabodi import Bodi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = load_dataset("rajpurkar/squad", split="train").shuffle(seed=1)
paras, pidx, qs = [], {}, []
for r in d:
    c = r["context"]
    if c not in pidx:
        if len(paras) >= 300:
            continue
        pidx[c] = len(paras); paras.append(c)
    qs.append((r["question"], pidx[c]))
qs = random.Random(1).sample(qs, 600)
variants = {"clean": [q for q, _ in qs], "keywords_typo": [typo_all(keywords(q), random.Random(i)) for i, (q, _) in enumerate(qs)]}
gold = [p for _, p in qs]
res = {}
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    b = Bodi({"dense_min_similarity": thr, "embedder_dir": os.path.join(ROOT, "models", "minilm")})
    b.ingest_batch([{"text": p, "format": "text", "source": "p%d" % i} for i, p in enumerate(paras)])
    row = {}
    for v, ql in variants.items():
        ans = wrong = 0
        for q, g in zip(ql, gold):
            r = b.query(q, k=5)
            ps = [int(re.match(r"F_p_(\d+)_", h["id"]).group(1)) for h in r["hits"]] if r["matched"] else []
            ans += (r["matched"] and not r["handoff"] and g in ps); wrong += (r["matched"] and not r["handoff"] and g not in ps)
        row[v] = {"answered_correctly_without_handoff": round(ans / len(ql), 4), "confidently_wrong": round(wrong / len(ql), 4)}
    row["objective"] = round(sum(x["answered_correctly_without_handoff"] - x["confidently_wrong"] for x in row.values()) / 2, 4)
    res[str(thr)] = row
    print(thr, row, flush=True)
best = max(res, key=lambda t: res[t]["objective"])
res["best"] = float(best)
json.dump(res, open(os.path.join(ROOT, "research", "results", "tune_hybrid.json"), "w"), indent=1)
print("best", best)
