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
"""
import json, os, sys, time
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
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    tr = train_suites(2000, 300)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.update({"items": "seeded shuffle positions 500..999 (bench.json uses 0..499)", "gated_settings": gb,
                "agree_settings": {k: ab[k] for k in ("agree", "trust", "mode")},
                "disclosure": "boolq positions 500..999 are also tune_grounding.py's dev set; no experience arm uses grounding", "suites": res.get("suites", {})})
    S_all = build_suites(1000, set(SUITES))
    for name in SUITES:
        if name in res["suites"]:
            continue
        S0 = S_all[name]
        assert S0["gold"][:500] == base[name]["gold"], "shuffle prefix differs from bench.json"
        S = dict(S0, states=S0["states"][500:1000], gold=S0["gold"][500:1000])
        gold = S["gold"]
        T = tr[name]; qid, qd = next(iter(T["q"].items()))
        assert qd == S["q"] and qid == S["qid"]
        r = {"n": len(gold)}
        lr, _, dropped = laya_rows(agent, S)
        lp = [None if p is None else int(np.argmax(p)) for _, p in lr]
        cl = [p == g for p, g in zip(lp, gold)]
        r["laya_torch"] = {"accuracy": round(float(np.mean(cl)), 4), "accuracy_ci95": wilson(sum(cl), len(cl)), "dropped": dropped, "pred": lp}
        b.forget()
        learned = b.learn([T["st"](x) for x in T["mem"]], T["q"], [{qid: T["y"](x)} for x in T["mem"]])
        corr = {}
        for tag, opts in (("gated", gated), ("gated_agree", agree)):
            t0 = time.time()
            out = b.decide_batch(S["states"], {qid: S["q"]}, **opts)
            el = time.time() - t0
            pred, over = [], 0
            for o in out:
                a = o["answers"][qid]
                over += bool(a.get("bodi", {}).get("experience", {}).get("override"))
                if a.get("bodi", {}).get("strategy") == "script_gate":
                    pred.append(None); continue
                p = [1 - a["noul"], a["noul"]] if a["type"] == "noul" else [a["probabilities"][l] for l in S["labels"]]
                pred.append(int(np.argmax(p)))
            corr[tag] = [p == g for p, g in zip(pred, gold)]
            r[tag] = {"accuracy": round(float(np.mean(corr[tag])), 4), "accuracy_ci95": wilson(sum(corr[tag]), len(gold)),
                      "overrides": over, "seconds": round(el, 1), "mcnemar_vs_laya_torch": mcnemar(corr[tag], cl), "pred": pred}
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
        r["learn"] = learned
        r["gold"] = gold
        res["suites"][name] = r
        print(name, {k: (v["accuracy"] if isinstance(v, dict) and "accuracy" in v else v) for k, v in r.items() if k not in ("gold", "learn")}, flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
