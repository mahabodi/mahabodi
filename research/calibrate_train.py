"""Run the product's calibrate() (learn(..., calibrate=200)) per suite on the SAME memory the tuning
uses (tune_experience train[seed2][:2000]): Laya's accuracy on 200 evenly strided STORED cases and the
memory's leave-one-out accuracy / k / T. Training data only - no validation or test items.
Feeds tune_memory_first.py.

    .venv/bin/python research/calibrate_train.py
"""
import json, os, sys
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_experience import suites  # noqa: E402
from mahabodi import Bodi  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "results", "calibration_train.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
SS = suites(2000, 300)
b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
for name in ["ag_news", "emotion", "banking77", "sst5", "prompt_injections", "boolq"]:
    if name in res:
        continue
    S = SS[name]; qid = next(iter(S["q"]))
    b.forget()
    r = b.learn([S["st"](x) for x in S["mem"]], S["q"], [{qid: S["y"](x)} for x in S["mem"]], calibrate=200)
    res[name] = r["calibration"][qid]
    print(name, res[name], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
