"""Does memory grounding help System-1 decisions? A/B/C on BoolQ (google/boolq validation, the same
500 seeded items as bench.py), laya English checkpoint, cache off.

  A  question only:   state {"question"}, "Is the answer to `question` yes?"
  B  memory-grounded: every item's passage is ingested into MahaBodi memory (hybrid retrieval,
     ingest_batch); decide_with_memory(state={"question"}, query=question) retrieves context and
     adds it as `memory`; "Based on `memory`, is the answer to `question` yes?"
     If retrieval hands off, no memory is added (by design).
  C  oracle passage:  bench.py's boolq setting (the right passage given directly).

Reported: accuracy with Wilson CI; confidently-wrong rate (confidence >= 0.8 and wrong); retrieval
hit rate (the item's own passage among the retrieved hits); B's accuracy split by hit / miss /
handoff; exact McNemar B vs A and B vs C. Retrieval that returns the WRONG passage is counted
and shown separately - that is the "grounded on the wrong memory" case.

    .venv/bin/python research/bench_grounding.py --out research/results/bench_grounding.json

v2: --style-file research/results/tune_grounding.json uses the grounding format chosen on the
disjoint dev sample (tune_grounding.py) for arm B; A and C are unchanged.

    .venv/bin/python research/bench_grounding.py --style-file research/results/tune_grounding.json \
        --out research/results/bench_grounding_v2.json
"""
import argparse, json, os, re, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites, mcnemar, wilson  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--style-file", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "bench_grounding.json"))
    a = ap.parse_args()
    S = build_suites(a.n, {"boolq"})["boolq"]
    gold = S["gold"]
    b = Bodi({"embedder_dir": os.path.join(ROOT, "models", "minilm")})
    b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    t0 = time.time()
    b.ingest_batch([{"text": st["passage"], "format": "text", "source": "q%d" % i} for i, st in enumerate(S["states"])])
    res = {"n": len(gold), "ingest_seconds": round(time.time() - t0, 1), "memory": b.stats()}
    qa = {"answer": {"type": "noul", "instructions": "Is the answer to `question` yes?"}}
    qb = {"answer": {"type": "noul", "instructions": "Based on `memory`, is the answer to `question` yes?"}}
    style = None
    if a.style_file:
        t = json.load(open(a.style_file))
        style = t["best_style"]
        qb = {"answer": {"type": "noul", "instructions": t["best_instructions"]}}
        res["grounding_format"] = {"from": os.path.basename(a.style_file), "best": t["best"], "style": style,
                                   "instructions": t["best_instructions"]}
    arms = {"A_question_only": [], "B_memory_grounded": [], "C_oracle_passage": []}
    meta = []
    for i, st in enumerate(S["states"]):
        q = {"question": st["question"]}
        arms["A_question_only"].append(b.decide(q, qa, cache=False)["answers"]["answer"])
        r = b.decide_with_memory(q, qb, query=st["question"], k=3, max_chars=1500, style=style, cache=False)
        arms["B_memory_grounded"].append(r["answers"]["answer"])
        hits = [int(m.group(1)) for h in r["memory"]["hits"] for m in [re.match(r"F_q_(\d+)_", h)] if m]
        meta.append({"used": r["memory"]["used"], "handoff": r["memory"]["handoff"], "stage": r["memory"]["stage"],
                     "hit": i in hits, "top": hits[:3]})
        arms["C_oracle_passage"].append(b.decide(st, {"answer": S["q"]}, cache=False)["answers"]["answer"])
    out = {}
    for name, ans in arms.items():
        pred = [int(x["noul"] >= 0.5) for x in ans]
        conf = [x["confidence"] for x in ans]
        corr = [p == g for p, g in zip(pred, gold)]
        out[name] = {"accuracy": round(float(np.mean(corr)), 4), "accuracy_ci95": wilson(sum(corr), len(corr)),
                     "confidently_wrong_rate": round(float(np.mean([(c >= 0.8) and not ok for c, ok in zip(conf, corr)])), 4),
                     "pred": pred, "confidence": conf}
    cb = [p == g for p, g in zip(out["B_memory_grounded"]["pred"], gold)]
    out["mcnemar_B_vs_A"] = mcnemar(cb, [p == g for p, g in zip(out["A_question_only"]["pred"], gold)])
    out["mcnemar_B_vs_C"] = mcnemar(cb, [p == g for p, g in zip(out["C_oracle_passage"]["pred"], gold)])
    yes = [g == 1 for g in gold]
    out["always_yes_accuracy"] = round(float(np.mean(yes)), 4)
    out["mcnemar_B_vs_always_yes"] = mcnemar(cb, yes)
    split = {}
    for tag, sel in (("retrieval_hit", lambda m: m["used"] and m["hit"]), ("retrieval_wrong_passage", lambda m: m["used"] and not m["hit"]),
                     ("handoff_no_memory", lambda m: not m["used"])):
        idx = [i for i, m in enumerate(meta) if sel(m)]
        split[tag] = {"n": len(idx), "B_accuracy": round(float(np.mean([cb[i] for i in idx])), 4) if idx else None,
                      "A_accuracy": round(float(np.mean([out["A_question_only"]["pred"][i] == gold[i] for i in idx])), 4) if idx else None}
    out["B_split_by_retrieval"] = split
    out["retrieval_hit_rate"] = round(float(np.mean([m["hit"] for m in meta])), 4)
    res.update(out); res["retrieval"] = meta; res["gold"] = gold
    for k in ("A_question_only", "B_memory_grounded", "C_oracle_passage"):
        print(k, {kk: v for kk, v in out[k].items() if kk not in ("pred", "confidence")})
    print("B vs A", out["mcnemar_B_vs_A"], "B vs C", out["mcnemar_B_vs_C"], "hit rate", out["retrieval_hit_rate"], split)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
