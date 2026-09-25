"""FRESH-sample test of experience memory: items 500..999 of the SAME seeded shuffles bench.py uses
(bench.json holds items 0..499; these are disjoint by construction and have never been scored).
Why fresh: the agreement override (tune_agree_gate.json) was designed AFTER bench_experience_gated.json
showed its gaps vs a kNN baseline, so testing it on those same 500 items would not be clean.

prompt_injections is excluded: its test split has only 116 items, all already used.
Disclosure: for boolq, positions 500..999 of the validation shuffle are ALSO tune_grounding.py's dev
set (grounding-format choice). No leak: no experience arm uses grounding or that choice.

Arms (same memory = tune_experience train[seed2][:2000], same MiniLM embedder):
  laya_torch   Laya's PyTorch model through Laya's own score_cases (no memory)
  gated        MahaBodi margin-gated experience (tune_margin_gate.json best) - the current default
  gated_agree  gated + agreement override (tune_agree_gate.json best)
  knn_own      memory alone: kNN with its OWN validation-tuned k / temperature (tune_knn_only.json)
Exact McNemar for every arm vs laya_torch, and gated_agree vs gated and vs knn_own.

    .venv/bin/python research/bench_fresh_experience.py

Third fresh sample (memory-first, learn(calibrate=200)): positions 1000..1499, never used by any
earlier run or design decision; adds arm `calibrated` = the REAL product: learn(..., calibrate=200)
then decide() with all defaults (gate + override + memory-first where calibration selects it).

    .venv/bin/python research/bench_fresh_experience.py --start 1000 --calibrate 200 \
        --out research/results/bench_fresh3_experience.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites, laya_rows, mcnemar, summarize, wilson  # noqa: E402
from tune_experience import suites as train_suites, label_idx, plain_text  # noqa: E402
from laya import Agent  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
OUT = os.path.join(R, "bench_fresh_experience.json")
SUITES = ["ag_news", "emotion", "banking77", "sst5", "boolq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--calibrate", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    assert a.start >= 500, "positions 0..499 are bench.json's test items"
    assert not (a.start != 500 and os.path.abspath(a.out) == OUT), "would overwrite the verified 500..999 results; pass --out"
    torch.set_num_threads(8)
    gb = json.load(open(os.path.join(R, "tune_margin_gate.json")))["best"]
    ab = json.load(open(os.path.join(R, "tune_agree_gate.json")))["best"]
    assert ab["mode"] == "vote"
    kown = json.load(open(os.path.join(R, "tune_knn_only.json")))
    base = json.load(open(os.path.join(R, "bench.json")))["suites"]
    gated = dict(cache=False, round_probabilities=False, experience_k=gb["k"], experience_temperature=gb["temperature"],
                 experience_weight=gb["weight"], experience_below_margin=gb["tau"])
    agree = dict(gated, experience_override_agree=ab["agree"], experience_override_min_trust=ab["trust"])
    agent = Agent("convaiinnovations/laya", device="cpu")
    import hashlib, mahabodi
    so = [os.path.join(os.path.dirname(mahabodi.__file__), f) for f in os.listdir(os.path.dirname(mahabodi.__file__)) if f.endswith(".so")]
    b = Bodi()
    defaults = b.decide_defaults()
    if a.calibrate:
        assert abs(defaults["experience_memory_first_margin"] - 0.2) < 1e-12, defaults
    prov = {"module": mahabodi.__file__, "native": {p: {"sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))} for p in so},
            "decide_defaults": defaults, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("provenance", json.dumps(prov), flush=True)
    b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    tr = train_suites(2000, 300)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    lo, hi = a.start, a.start + 500
    res.update({"items": "seeded shuffle positions %d..%d (bench.json uses 0..499%s)" % (lo, hi - 1, "; 500..999 used by bench_fresh_experience.json" if lo >= 1000 else ""),
                "calibrate": a.calibrate, "provenance": prov, "gated_settings": gb,
                "agree_settings": {k: ab[k] for k in ("agree", "trust", "mode")},
                "disclosure": ("boolq positions 500..999 are also tune_grounding.py's dev set; no experience arm uses grounding" if lo < 1000 else ""), "suites": res.get("suites", {})})
    S_all = build_suites(hi, set(SUITES))
    npz = os.path.join(ROOT, "research", "cache", "tune_text_%s_m2000_v300.npz")
    for name in SUITES:
        if name in res["suites"]:
            continue
        S0 = S_all[name]
        assert S0["gold"][:500] == base[name]["gold"], "shuffle prefix differs from bench.json"
        assert len(S0["gold"]) >= hi, "%s has only %d items" % (name, len(S0["gold"]))
        S = dict(S0, states=S0["states"][lo:hi], gold=S0["gold"][lo:hi])
        gold = S["gold"]
        T = tr[name]; qid, qd = next(iter(T["q"].items()))
        assert qd == S["q"] and qid == S["qid"]
        r = {"n": len(gold)}
        # the stored memory must be the same items the tuning cache (and calibration) describes
        z = np.load(npz % name)
        chk = np.array(b.embed_text([plain_text(T["st"](x)) for x in list(T["mem"])[:32]]), dtype=np.float32)
        chk /= np.linalg.norm(chk, axis=1, keepdims=True)
        mv = z["Mv"][:32] / np.linalg.norm(z["Mv"][:32], axis=1, keepdims=True)
        assert np.min(np.sum(chk * mv, 1)) > 0.999 and len(z["My"]) == len(T["mem"]), "memory differs from the tuning cache"
        lr, _, dropped = laya_rows(agent, S)
        lp = [None if p is None else int(np.argmax(p)) for _, p in lr]
        cl = [p == g for p, g in zip(lp, gold)]
        r["laya_torch"] = {"accuracy": round(float(np.mean(cl)), 4), "accuracy_ci95": wilson(sum(cl), len(cl)), "dropped": dropped, "pred": lp}
        b.forget()
        learned = b.learn([T["st"](x) for x in T["mem"]], T["q"], [{qid: T["y"](x)} for x in T["mem"]], calibrate=a.calibrate)
        corr = {}
        arms = [("gated", dict(gated, experience_override_agree=0, experience_memory_first_margin=-1.0)),
                ("gated_agree", dict(agree, experience_memory_first_margin=-1.0))]
        if a.calibrate:
            arms.append(("calibrated", dict(cache=False, round_probabilities=False)))  # product defaults
        for tag, opts in arms:
            t0 = time.time()
            out = b.decide_batch(S["states"], {qid: S["q"]}, **opts)
            el = time.time() - t0
            pred, over, mf = [], 0, 0
            for o in out:
                ans = o["answers"][qid]
                over += bool(ans.get("bodi", {}).get("experience", {}).get("override"))
                mf += bool(ans.get("bodi", {}).get("experience", {}).get("memory_first"))
                if ans.get("bodi", {}).get("strategy") == "script_gate":
                    pred.append(None); continue
                p = [1 - ans["noul"], ans["noul"]] if ans["type"] == "noul" else [ans["probabilities"][l] for l in S["labels"]]
                pred.append(int(np.argmax(p)))
            corr[tag] = [p == g for p, g in zip(pred, gold)]
            r[tag] = {"accuracy": round(float(np.mean(corr[tag])), 4), "accuracy_ci95": wilson(sum(corr[tag]), len(gold)),
                      "overrides": over, "memory_first": mf, "seconds": round(el, 1), "mcnemar_vs_laya_torch": mcnemar(corr[tag], cl), "pred": pred}
        own = kown[name]
        Mv = np.array(b.embed_text([plain_text(T["st"](x)) for x in T["mem"]]), dtype=np.float32)
        My = np.array([label_idx(qd, T["y"](x)) for x in T["mem"]])
        E = np.array(b.embed_text([plain_text(s) for s in S["states"]]), dtype=np.float32)
        sims = E @ Mv.T
        top = np.argsort(-sims, 1)[:, :own["k"]]
        n = 2 if qd["type"] == "noul" else len(qd["criteria"])
        kp = [int(np.bincount(My[top[i]], weights=np.exp((sims[i, top[i]] - sims[i, top[i]].max()) / own["temperature"]), minlength=n).argmax())
              for i in range(len(E))]
        corr["knn_own"] = [p == g for p, g in zip(kp, gold)]
        r["knn_own"] = {"k": own["k"], "temperature": own["temperature"], "accuracy": round(float(np.mean(corr["knn_own"])), 4),
                        "accuracy_ci95": wilson(sum(corr["knn_own"]), len(gold)), "mcnemar_vs_laya_torch": mcnemar(corr["knn_own"], cl), "pred": kp}
        r["mcnemar_gated_agree_vs_gated"] = mcnemar(corr["gated_agree"], corr["gated"])
        r["mcnemar_gated_agree_vs_knn_own"] = mcnemar(corr["gated_agree"], corr["knn_own"])
        r["mcnemar_gated_vs_knn_own"] = mcnemar(corr["gated"], corr["knn_own"])
        if "calibrated" in corr:
            r["mcnemar_calibrated_vs_knn_own"] = mcnemar(corr["calibrated"], corr["knn_own"])
            r["mcnemar_calibrated_vs_gated_agree"] = mcnemar(corr["calibrated"], corr["gated_agree"])
            r["calibrated"]["mcnemar_vs_laya_torch"] = mcnemar(corr["calibrated"], cl)
        r["learn"] = learned
        r["gold"] = gold
        res["suites"][name] = r
        print(name, {k: (v["accuracy"] if isinstance(v, dict) and "accuracy" in v else v) for k, v in r.items() if k not in ("gold", "learn")}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
