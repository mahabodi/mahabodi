"""USECASES #2: intent routing with out-of-scope (CLINC150 "plus": 150 intents + oos).
Pre-registered in research/USECASES.md. English laya checkpoint (models/laya-v2), zero-shot.

  A  Laya: one choice over all 150 intent names (MahaBodi laya_compatible: identical maths to Laya,
     verified 0 discordance) + OOS when top probability < tau_A
  B  Laya + MiniLM shortlist: laya.shortlist.shortlist_choice (Laya's own code, k=20) with the
     all-MiniLM-L6-v2 bi-encoder, then Laya on the 20 + OOS when top probability < tau_B
  C  MahaBodi defaults (tournament) + OOS when top probability < tau_C

Each tau is tuned on a seeded VALIDATION sample (maximising overall accuracy, OOS as a class);
then ONE seeded test run. Metrics: overall accuracy (151 classes), in-scope accuracy, OOS recall,
OOS precision; exact McNemar on overall correctness, C vs B (the win condition) and C vs A.

    .venv/bin/python research/bench_clinc.py --val 600 --test 1000
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import LAYA_COMMIT, mcnemar, wilson  # noqa: E402
from datasets import load_dataset  # noqa: E402
from laya.shortlist import shortlist_choice  # noqa: E402
from mahabodi import Bodi  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INS = "Which intent does `utterance` express?"


def load(split, n, seed):
    d = load_dataset("clinc_oos", "plus", split=split).shuffle(seed=seed)
    return d.select(range(min(n, len(d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "bench_clinc.json"))
    a = ap.parse_args()
    names = load_dataset("clinc_oos", "plus", split="validation").features["intent"].names
    oos = names.index("oos")
    intents = [i for i in range(len(names)) if i != oos]
    labels = [names[i].replace("_", " ") for i in intents]
    crit = {l: None for l in labels}
    q = {"intent": {"type": "choice", "instructions": INS, "criteria": crit}}
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    ef = lambda texts: st.encode(list(texts), batch_size=64, convert_to_numpy=True)
    compat = dict(permute=False, max_options_per_pass=0, script_gate=False, cache=False, experience_k=0, round_probabilities=False)

    def run(ds):
        rows = {"A": [], "B": [], "C": []}
        for r in ds:
            s = {"utterance": r["text"]}
            pa = b.decide(s, q, **compat)["answers"]["intent"]["probabilities"]
            rows["A"].append((max(pa, key=pa.get), max(pa.values())))
            short = shortlist_choice(s, crit, ef, k=20, instructions=INS)
            qb = {"intent": {"type": "choice", "instructions": INS, "criteria": {l: None for l in short}}}
            pb = b.decide(s, qb, **compat)["answers"]["intent"]["probabilities"]
            rows["B"].append((max(pb, key=pb.get), max(pb.values())))
            ac = b.decide(s, q, cache=False, experience_k=0, round_probabilities=False)["answers"]["intent"]
            pc = ac["probabilities"]
            rows["C"].append((max(pc, key=pc.get), max(pc.values())))
        gold = [names[r["intent"]].replace("_", " ") if r["intent"] != oos else "oos" for r in ds]
        return rows, gold

    def score(pred_conf, gold, tau):
        pred = ["oos" if c < tau else p for p, c in pred_conf]
        corr = [p == g for p, g in zip(pred, gold)]
        ins = [c for c, g in zip(corr, gold) if g != "oos"]
        oos_pred = [p == "oos" for p in pred]; oos_gold = [g == "oos" for g in gold]
        tp = sum(p and g for p, g in zip(oos_pred, oos_gold))
        return {"accuracy": round(float(np.mean(corr)), 4), "accuracy_ci95": wilson(sum(corr), len(corr)),
                "in_scope_accuracy": round(float(np.mean(ins)), 4) if ins else None,
                "oos_recall": round(tp / max(1, sum(oos_gold)), 4), "oos_precision": round(tp / max(1, sum(oos_pred)), 4)}, corr, pred

    t0 = time.time()
    vrows, vgold = run(load("validation", a.val, 7))
    taus = {}
    for arm in "ABC":
        grid = sorted({round(c, 3) for _, c in vrows[arm]} | {0.0, 1.01})
        taus[arm] = max(grid, key=lambda t: (score(vrows[arm], vgold, t)[0]["accuracy"], -t))
    res = {"laya_commit": LAYA_COMMIT, "val_n": len(vgold), "taus_from_validation": taus,
           "val": {arm: score(vrows[arm], vgold, taus[arm])[0] for arm in "ABC"}}
    print("val", res["val"], "taus", taus, flush=True)
    trows, tgold = run(load("test", a.test, 7))
    res["test_n"] = len(tgold)
    corrs = {}
    for arm in "ABC":
        m, corrs[arm], pred = score(trows[arm], tgold, taus[arm])
        res[arm] = {**m, "pred": pred}
        print(arm, m, flush=True)
    res["mcnemar_C_vs_B"] = mcnemar(corrs["C"], corrs["B"])
    res["mcnemar_C_vs_A"] = mcnemar(corrs["C"], corrs["A"])
    res["mcnemar_B_vs_A"] = mcnemar(corrs["B"], corrs["A"])
    res["gold"] = tgold
    res["seconds"] = round(time.time() - t0, 1)
    print("C vs B", res["mcnemar_C_vs_B"], "C vs A", res["mcnemar_C_vs_A"], flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
