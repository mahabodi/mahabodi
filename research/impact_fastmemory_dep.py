"""fastmemory-dep impact check (DEV only): does swapping the vendored clustering for the fastmemory crate change
grounded decisions?

Run mode (`--tag <name>`): the BoolQ DEV items of tune_grounding.py (validation shuffle(seed=bench.SEED)[500:1000]
ingested; the first --q asked), decide_with_memory with the default labelled style (k = 3, max_chars = 1500 — the
style whose "[block] related:" lines depend on the clustering), cache off. Per item it records the decision, the
confidence, the memory flags and the EXACT rendered context string (`context`, the same call the grounding uses).
Plus `stats()` (nodes, edges, blocks) after ingest. Output: research/results/impact_fmdep_<tag>.json.

Run it once per build, same items, e.g.:
    --tag vendored      binding built from main (vendored louvain.rs)
    --tag dep_native    binding built from the fastmemory-dep branch, FASTMEMORY_NATIVE_LIB set
    --tag dep_inline    the same build, FASTMEMORY_CLUSTER=builtin

Compare mode (`--compare A.json B.json`): blocks delta, items whose context string changed, items whose decision
changed, accuracy per side with Wilson CIs and exact McNemar.
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
QMEM = {"answer": {"type": "noul", "instructions": "Based on `memory`, is the answer to `question` yes?"}}


def run(a):
    from bench import SEED, wilson  # noqa: E402  (torch-light import path as tune_grounding)
    from datasets import load_dataset
    from mahabodi import Bodi
    d = load_dataset("google/boolq", split="validation").shuffle(seed=SEED).select(range(500, 1000))
    b = Bodi({"embedder_dir": os.path.join(ROOT, "models", "minilm")})
    b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    t0 = time.time()
    b.ingest_batch([{"text": r["passage"], "format": "text", "source": "q%d" % i} for i, r in enumerate(d)])
    rows = list(d)[:a.q]
    gold = [int(bool(r["answer"])) for r in rows]
    out = {"tag": a.tag, "dev_items": "boolq validation shuffle(seed=%d)[500:1000], asked first %d" % (SEED, len(rows)),
           "style": "default labelled (block/sibling context), k=3, max_chars=1500, cache off",
           "ingest_seconds": round(time.time() - t0, 1), "stats": b.stats(),
           "clustering": b.stats().get("clustering"), "provenance": provenance(), "items": [], "gold": gold}
    for i, r0 in enumerate(rows):
        q = r0["question"]
        ctx = b.context(q, k=3, max_chars=1500)
        rr = b.decide_with_memory({"question": q}, QMEM, query=q, k=3, max_chars=1500, cache=False)
        ans = rr["answers"]["answer"]
        out["items"].append({"i": i, "pred": int(ans["noul"] >= 0.5), "noul": round(float(ans["noul"]), 6),
                             "confidence": round(float(ans["confidence"]), 6),
                             "context_sha256": hashlib.sha256(ctx["context"].encode()).hexdigest(),
                             "context": ctx["context"], "stage": rr["memory"]["stage"],
                             "used": rr["memory"]["used"], "handoff": rr["memory"]["handoff"],
                             "hits": rr["memory"]["hits"][:3]})
        if i % 50 == 0:
            print(a.tag, i, "/", len(rows), flush=True)
    corr = [it["pred"] == g for it, g in zip(out["items"], gold)]
    out["accuracy"] = round(float(np.mean(corr)), 4)
    out["accuracy_ci95"] = wilson(sum(corr), len(corr))
    path = os.path.join(R, "impact_fmdep_%s.json" % a.tag)
    json.dump(out, open(path, "w"))
    print("DONE", a.tag, "accuracy", out["accuracy"], "blocks", (out["stats"] or {}).get("blocks"), "->", path, flush=True)


def compare(pa, pb):
    from bench import wilson, mcnemar  # noqa: E402
    A, B = json.load(open(pa)), json.load(open(pb))
    assert A["gold"] == B["gold"] and len(A["items"]) == len(B["items"]), "different item sets"
    gold = A["gold"]
    ca = [it["pred"] == g for it, g in zip(A["items"], gold)]
    cb = [it["pred"] == g for it, g in zip(B["items"], gold)]
    ctx_diff = [i for i, (x, y) in enumerate(zip(A["items"], B["items"])) if x["context_sha256"] != y["context_sha256"]]
    pred_diff = [i for i, (x, y) in enumerate(zip(A["items"], B["items"])) if x["pred"] != y["pred"]]
    hits_diff = [i for i, (x, y) in enumerate(zip(A["items"], B["items"])) if x["hits"] != y["hits"]]
    rep = {"a": A["tag"], "b": B["tag"],
           "blocks": {"a": (A.get("stats") or {}).get("blocks"), "b": (B.get("stats") or {}).get("blocks")},
           "clustering": {"a": A.get("clustering"), "b": B.get("clustering")},
           "n": len(gold),
           "context_changed": {"n": len(ctx_diff), "items": ctx_diff[:50]},
           "hits_changed": {"n": len(hits_diff), "items": hits_diff[:50]},
           "pred_changed": {"n": len(pred_diff), "items": pred_diff},
           "accuracy": {"a": round(float(np.mean(ca)), 4), "a_ci95": wilson(sum(ca), len(ca)),
                        "b": round(float(np.mean(cb)), 4), "b_ci95": wilson(sum(cb), len(cb))},
           "mcnemar_a_vs_b": mcnemar(ca, cb)}
    path = os.path.join(R, "impact_fmdep_compare_%s_vs_%s.json" % (A["tag"], B["tag"]))
    json.dump(rep, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ()}, indent=1)[:1200])
    print("->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--q", type=int, default=300)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        assert a.tag, "--tag or --compare required"
        run(a)


if __name__ == "__main__":
    main()
