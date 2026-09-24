"""Does the margin-gated experience setting (tune_margin_gate.json best) behave the same through the
REAL decide path as in tune_margin_gate.py's numpy simulation? Validation items only
(tune_experience.suites: memory train[seed2][:2000], val [2000:2300]). Laya = decide with experience off.

    .venv/bin/python research/check_gate_real.py
"""
import json, os, sys
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import suites  # noqa: E402
from bench import mcnemar  # noqa: E402
from mahabodi import Bodi  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gb = json.load(open(os.path.join(ROOT, "research", "results", "tune_margin_gate.json")))["best"]
names = sys.argv[1:] or ["emotion", "ag_news"]
SS = suites(2000, 300)
res = {}
for name in names:
    S = SS[name]; qid = next(iter(S["q"]))
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    b.learn([S["st"](r) for r in S["mem"]], S["q"], [{qid: S["y"](r)} for r in S["mem"]])
    val = [S["st"](r) for r in S["val"]]; Y = [S["y"](r) for r in S["val"]]
    lay = b.decide_batch(val, S["q"], cache=False, experience_k=0)
    mem = b.decide_batch(val, S["q"], cache=False, experience_k=gb["k"], experience_temperature=gb["temperature"],
                         experience_weight=gb["weight"], experience_below_margin=gb["tau"])
    a = [o["answers"][qid]["choice"] == y for o, y in zip(mem, Y)]; l = [o["answers"][qid]["choice"] == y for o, y in zip(lay, Y)]
    res[name] = {"real_mem": round(sum(a) / len(a), 4), "real_laya": round(sum(l) / len(l), 4), "mcnemar": mcnemar(a, l),
                 "sim": gb["per_suite"].get(name)}
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(ROOT, "research", "results", "check_gate_real.json"), "w"), indent=1)
