"""Tune Bodi's decision options on VALIDATION data only (never on the test items bench.py reports).

  ag_news   -> seeded sample of the TRAIN split (ag_news has no validation split)
  emotion   -> seeded sample of the VALIDATION split
  banking77 -> seeded sample of the TRAIN split

Adaptive ensemble: each case is scored once in caller order and once reversed (reversing
the criteria dict yields exactly the token sequence of option_order=reversed), then every
threshold is simulated offline. Tournament configs for banking77 are run for real.

    .venv/bin/python research/tune.py --n 300 --out research/results/tune.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
from datasets import load_dataset
from mahabodi import Bodi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = {"world": "world news and international politics", "sports": "sports", "business": "business and economy", "sci_tech": "science and technology"}
EMO = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def conf(p):
    p = np.asarray(p); k = len(p)
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return 1 - ent / np.log(k)


def suites(n):
    ag = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=1).select(range(n))
    em = load_dataset("dair-ai/emotion", "split", split="validation").shuffle(seed=1).select(range(n))
    bk = load_dataset("mteb/banking77", split="train").shuffle(seed=1).select(range(n))
    labels = sorted(set(load_dataset("mteb/banking77", split="test")["label_text"]))
    lab = [x.replace("_", " ") for x in labels]
    return {
        "ag_news": ([{"article": r["text"]} for r in ag], [r["label"] for r in ag], "topic", "What is the topic of `article`?", dict(AG)),
        "emotion": ([{"text": r["text"]} for r in em], [r["label"] for r in em], "emotion", "Which emotion is most strongly expressed in `text`?", {x: None for x in EMO}),
        "banking77": ([{"message": r["text"]} for r in bk], [lab.index(r["label_text"].replace("_", " ")) for r in bk], "intent", "Which banking intent does `message` express?", {x: None for x in lab}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "tune.json"))
    ap.add_argument("--only", default="", help="comma list of sections to (re)run; others are kept from --out")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    compat = dict(permute=False, max_options_per_pass=0, script_gate=False, cache=False)
    out = {"n": a.n, "splits": {"ag_news": "train seed=1", "emotion": "validation seed=1", "banking77": "train seed=1"}}
    prev = json.load(open(a.out)) if (only and os.path.exists(a.out)) else {}
    out.update({k: v for k, v in prev.items() if k not in only})
    S = suites(a.n)
    for name in ("ag_news", "emotion"):
        if only and name not in only:
            continue
        states, gold, qid, ins, crit = S[name]
        labels = list(crit)
        fwd = b.decide_batch(states, {qid: {"type": "choice", "instructions": ins, "criteria": crit}}, **compat)
        rev = b.decide_batch(states, {qid: {"type": "choice", "instructions": ins, "criteria": dict(reversed(list(crit.items())))}}, **compat)
        P1 = np.array([[r["answers"][qid]["probabilities"][l] for l in labels] for r in fwd])
        P2 = np.array([[r["answers"][qid]["probabilities"][l] for l in labels] for r in rev])
        g = np.array(gold)
        res = {}
        for tau in [None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0]:
            if tau is None:
                P, extra = P1, 0.0
            else:
                use = np.array([tau == 0.0 or conf(p) < tau for p in P1])
                P = np.where(use[:, None], (P1 + P2) / 2, P1); extra = float(use.mean())
            res["off" if tau is None else ("always" if tau == 0.0 else "tau=%.1f" % tau)] = {
                "accuracy": round(float((P.argmax(1) == g).mean()), 4), "rows_per_decision": round(1 + extra, 3)}
        out[name] = res
        print(name, json.dumps(res), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    states, gold, qid, ins, crit = S["banking77"]
    labels = list(crit)
    q = {qid: {"type": "choice", "instructions": ins, "criteria": crit}}
    res = dict(prev.get("banking77", {}))  # resume: finished configs are not re-run
    for mo, fin, perm in [(0, 3, False), (16, 3, False), (20, 4, False), (12, 2, False), (24, 4, False), (16, 5, False), (16, 3, True), (20, 4, True), (24, 4, True)]:
        key0 = "laya_compat" if mo == 0 else "max%d_fin%d_perm%d" % (mo, fin, perm)
        if key0 in res:
            continue
        s0 = b.stats()["system1"]["rows_scored"]
        t = time.time()
        rr = b.decide_batch(states, q, permute=perm, max_options_per_pass=mo, finalists_per_group=fin, script_gate=False, cache=False)
        el = time.time() - t
        pred = [labels.index(r["answers"][qid]["choice"]) for r in rr]
        key = "laya_compat" if mo == 0 else "max%d_fin%d_perm%d" % (mo, fin, perm)
        res[key] = {"accuracy": round(float(np.mean([p == x for p, x in zip(pred, gold)])), 4),
                    "rows_per_decision": round((b.stats()["system1"]["rows_scored"] - s0) / len(states), 2), "seconds": round(el, 1)}
        print("banking77", key, res[key], flush=True)
        out["banking77"] = res
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
