"""Choose the grounding format for decide_with_memory on a DEV sample disjoint from the test items.

bench_grounding.py tests on the first 500 items of BoolQ validation shuffled with bench.SEED. This
script uses items 500..999 of the SAME shuffle (disjoint by construction): their passages are
ingested exactly as in the test (ingest_batch, hybrid retrieval), and the first --q of them are asked.
Formats compared (all k=3 retrieval, max_chars 1500):

  labelled   current default: "- label: text" lines + "[block] related:" lines under `memory`, last
  p1 / p2 / p3  passages_only, top 1/2/3 passage texts, under `passage`, placed first;
             instruction "Based on `passage`, is the answer to `question` yes?" (Laya's own format)

The best format by dev accuracy (ties -> fewer passages) is written to the output; bench_grounding.py
--style-file then runs it ONCE on the test items.

    .venv/bin/python research/tune_grounding.py --out research/results/tune_grounding.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import SEED, wilson  # noqa: E402
from datasets import load_dataset  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QMEM = {"answer": {"type": "noul", "instructions": "Based on `memory`, is the answer to `question` yes?"}}
QPAS = {"answer": {"type": "noul", "instructions": "Based on `passage`, is the answer to `question` yes?"}}
FORMATS = {
    "labelled": (None, QMEM),
    "p1": ({"passages_only": True, "top": 1, "key": "passage", "first": True}, QPAS),
    "p2": ({"passages_only": True, "top": 2, "key": "passage", "first": True}, QPAS),
    "p3": ({"passages_only": True, "top": 3, "key": "passage", "first": True}, QPAS),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "tune_grounding.json"))
    a = ap.parse_args()
    d = load_dataset("google/boolq", split="validation").shuffle(seed=SEED).select(range(500, 1000))
    b = Bodi({"embedder_dir": os.path.join(ROOT, "models", "minilm")})
    b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    b.ingest_batch([{"text": r["passage"], "format": "text", "source": "q%d" % i} for i, r in enumerate(d)])
    rows = list(d)[:a.q]
    gold = [int(bool(r["answer"])) for r in rows]
    t0 = time.time()
    res = {"dev_items": "boolq validation shuffle(seed=%d)[500:1000], asked first %d" % (SEED, len(rows)), "formats": {}}
    for name, (style, q) in FORMATS.items():
        pred, conf = [], []
        for r in rows:
            x = b.decide_with_memory({"question": r["question"]}, q, query=r["question"], k=3, max_chars=1500,
                                     style=style, cache=False)["answers"]["answer"]
            pred.append(int(x["noul"] >= 0.5)); conf.append(x["confidence"])
        corr = [p == g for p, g in zip(pred, gold)]
        res["formats"][name] = {"style": style, "instructions": q["answer"]["instructions"],
                                "accuracy": round(float(np.mean(corr)), 4), "accuracy_ci95": wilson(sum(corr), len(corr)),
                                "confidently_wrong_rate": round(float(np.mean([(c >= 0.8) and not ok for c, ok in zip(conf, corr)])), 4)}
        print(name, {k: v for k, v in res["formats"][name].items() if k != "style"}, flush=True)
    order = list(FORMATS)
    best = max(order, key=lambda n: (res["formats"][n]["accuracy"], -order.index(n) if n == "labelled" else -int(n[1:])))
    res["best"] = best
    res["best_style"] = FORMATS[best][0]
    res["best_instructions"] = FORMATS[best][1]["answer"]["instructions"]
    res["seconds"] = round(time.time() - t0, 1)
    print("best", best, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
