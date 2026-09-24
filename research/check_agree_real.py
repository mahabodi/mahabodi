"""Real decide path vs tune_agree_gate.py simulation, VALIDATION items only (no test data).

    .venv/bin/python research/check_agree_real.py banking77 emotion
"""
import json, os, sys
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import suites  # noqa: E402
from mahabodi import Bodi  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
gb = json.load(open(os.path.join(R, "tune_margin_gate.json")))["best"]
ab = json.load(open(os.path.join(R, "tune_agree_gate.json")))
SS = suites(2000, 300)
out = {}
for name in sys.argv[1:]:
    S = SS[name]; qid = next(iter(S["q"]))
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    learned = b.learn([S["st"](r) for r in S["mem"]], S["q"], [{qid: S["y"](r)} for r in S["mem"]])
    val = [S["st"](r) for r in S["val"]]; Y = [S["y"](r) for r in S["val"]]
    o = b.decide_batch(val, S["q"], cache=False, experience_k=gb["k"], experience_temperature=gb["temperature"], experience_weight=gb["weight"],
                       experience_below_margin=gb["tau"], experience_override_agree=ab["best"]["agree"], experience_override_min_trust=ab["best"]["trust"])
    acc = sum(x["answers"][qid]["choice"] == y for x, y in zip(o, Y)) / len(Y)
    out[name] = {"real": round(acc, 4), "sim": ab["best"]["per_suite"][name], "sim_loo": ab["loo_trust"][name],
                 "overrides": sum(bool(x["answers"][qid]["bodi"].get("experience", {}).get("override")) for x in o)}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(R, "check_agree_real.json"), "w"), indent=1)
