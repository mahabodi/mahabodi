"""MahaBodi WITH EXPERIENCE MEMORY on the same seeded test items as research/bench.py.

This is a different setting from Laya's zero-shot numbers and is reported as its own column:
MahaBodi is given M labelled TRAIN examples as experience (no gradient training); Laya cannot
use them. Memory = the same train examples used in tuning (train[seed2][:M]; prompt_injections
train[seed2][100:]); test items are read back from bench.json and checked against the gold labels.

Settings per suite come ONLY from validation tuning:
  per_suite   tune_experience_text.json best (banking77: tune_experience_real.json best)
  global      tune_experience_text.json global_default

McNemar compares each against laya_torch's per-item predictions saved in bench.json.

    .venv/bin/python research/bench_experience.py --out research/results/bench_experience.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites, metrics, mcnemar, summarize  # noqa: E402
from tune_experience import suites as train_suites, label_idx  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")


def settings(tune_file, auto_trust):
    t = json.load(open(os.path.join(R, tune_file)))
    g = dict(t["global_default"], auto_trust=auto_trust)
    per = {n: dict(v["best"], auto_trust=auto_trust) for n, v in t["suites"].items()}
    real = os.path.join(R, "tune_experience_real.json")
    if os.path.exists(real):
        rr = json.load(open(real))
        best = max(rr.items(), key=lambda kv: kv[1]["accuracy"])[0]
        k, T, w, c = best.split("_")
        # validated through real decide WITHOUT the trust gate, so it runs without it
        per["banking77"] = {"k": int(k[1:]), "temperature": float(T[1:]), "weight": float(w[1:]), "candidates": int(c[1:]),
                            "auto_trust": False, "source": "tune_experience_real.json"}
    return g, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--memory", type=int, default=2000)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(R, "bench_experience.json"))
    ap.add_argument("--tune-file", default="tune_experience_text.json")
    ap.add_argument("--auto-trust", action="store_true")
    ap.add_argument("--gate-file", default="", help="tune_margin_gate.json: run only its validation-best gated global setting")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    base = json.load(open(os.path.join(R, "bench.json")))["suites"]
    g, per = settings(a.tune_file, a.auto_trust)
    if a.gate_file:
        gb = json.load(open(os.path.join(R, a.gate_file)))["best"]
        g = {"k": gb["k"], "temperature": gb["temperature"], "weight": gb["weight"], "below_margin": gb["tau"], "auto_trust": False,
             "source": a.gate_file}
        per = {}
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    tr = train_suites(a.memory, 300)
    res = json.load(open(a.out)) if (os.path.exists(a.out) and only) else {}
    res.update({"tune_file": a.tune_file, "auto_trust": a.auto_trust, "memory_per_suite": a.memory, "global_default": g, "per_suite_settings": per, "suites": res.get("suites", {})})
    for name, S in build_suites(a.n, only).items():
        if name not in base:
            print("skip", name, "(not in bench.json yet)"); continue
        assert base[name]["gold"] == S["gold"], "test items differ from bench.json"
        T = tr[name]; qid, qd = next(iter(T["q"].items()))
        assert qd == S["q"] and qid == S["qid"], "question definition differs from bench"
        b.forget()
        t0 = time.time()
        b.learn([T["st"](r) for r in T["mem"]], T["q"], [{qid: T["y"](r)} for r in T["mem"]])
        learn_s = time.time() - t0
        r = {"memory": len(T["mem"]), "learn_seconds": round(learn_s, 1)}
        lp = base[name]["laya_torch_pred"]
        cl = [p == gl for p, gl in zip(lp, S["gold"])]
        for tag, s in (([("per_suite", per[name])] if name in per else []) + [("global", g)]):
            opts = dict(cache=False, round_probabilities=False, experience_k=s["k"], experience_temperature=s["temperature"],
                        experience_weight=s["weight"], experience_candidates=s.get("candidates", 3),
                        experience_auto_trust=s.get("auto_trust", False), experience_below_margin=s.get("below_margin", 1.01))
            s0 = b.stats()["system1"]["rows_scored"]
            t0 = time.time()
            out = b.decide_batch(S["states"], {S["qid"]: S["q"]}, **opts)
            el = time.time() - t0
            rows = []
            for gl, o in zip(S["gold"], out):
                ans = o["answers"][S["qid"]]
                if ans.get("bodi", {}).get("strategy") == "script_gate":
                    rows.append((gl, [1.0 / len(S["labels"])] * len(S["labels"]))); continue
                rows.append((gl, [1 - ans["noul"], ans["noul"]] if ans["type"] == "noul" else [ans["probabilities"][l] for l in S["labels"]]))
            m = summarize(rows, el, {"rows_per_decision": round((b.stats()["system1"]["rows_scored"] - s0) / len(rows), 3), "settings": s})
            pred = [int(np.argmax(p)) for _, p in rows]
            m["mcnemar_vs_laya_torch"] = mcnemar([p == gl for p, gl in zip(pred, S["gold"])], cl)
            m["pred"] = pred
            r["bodi_experience_" + tag] = m
            print(name, tag, {k: v for k, v in m.items() if k not in ("pred",)}, flush=True)
        r["laya_torch_accuracy"] = base[name]["laya_torch"]["accuracy"]
        res["suites"][name] = r
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
