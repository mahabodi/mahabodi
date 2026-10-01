"""fastmemory retrieval before/after: the same SQuAD sample, queries and metrics as bench_retrieval.py.

The "before" is in retrieval.json (fastmemory's PyPI package returned any block for 0.0% of full questions and kept
no passage ids). The "after" here is fastmemory's passage search (`fastmemory.SearchMemory`, lexical cascade ported
from MahaBodi), next to MahaBodi's lexical `bodi` arm and BM25 on identical queries, with exact McNemar on recall@5.

    .venv/bin/python research/bench_retrieval_fastmemory.py --paragraphs 300 --questions 600
"""
import argparse, json, os, random, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_retrieval import typo, typo_all, keywords, toks, ROOT  # noqa: E402
from provenance import provenance  # noqa: E402
from store_parity_gate import mcnemar  # noqa: E402  (bench.py's, verbatim)


def score(name, run_query, variants, gold, res):
    for vname, qlist in variants.items():
        r1 = r5 = hand = wrong = answered = 0
        lat, items, stages = [], [], {}
        for q, g in zip(qlist, gold):
            t = time.perf_counter(); top, stage, matched, handoff = run_query(q); lat.append((time.perf_counter() - t) * 1000)
            r1 += bool(top and top[0] == g); r5 += g in top
            hand += handoff; wrong += (matched and not handoff and g not in top); answered += (matched and not handoff and g in top)
            stages[stage] = stages.get(stage, 0) + 1
            items.append({"q": q, "gold": g, "top": top, "stage": stage, "handoff": handoff})
        n = len(qlist)
        res["systems"].setdefault(name, {})[vname] = {
            "recall@1": round(r1 / n, 4), "recall@5": round(r5 / n, 4), "handoff_rate": round(hand / n, 4),
            "confidently_wrong_rate": round(wrong / n, 4), "answered_correctly_without_handoff": round(answered / n, 4),
            "stages": stages, "query_ms_p50": round(float(np.percentile(lat, 50)), 3), "items": items}
        print(name, vname, {k: v for k, v in res["systems"][name][vname].items() if k != "items"}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paragraphs", type=int, default=300)
    ap.add_argument("--questions", type=int, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import fastmemory
    from datasets import load_dataset
    from mahabodi import Bodi
    from rank_bm25 import BM25Okapi
    d = load_dataset("rajpurkar/squad", split="validation").shuffle(seed=0)
    paras, pidx, qs = [], {}, []
    for r in d:
        c = r["context"]
        if c not in pidx:
            if len(paras) >= a.paragraphs:
                continue
            pidx[c] = len(paras); paras.append(c)
        qs.append((r["question"], pidx[c], r["id"]))
    rng = random.Random(0)
    qs = rng.sample(qs, min(a.questions, len(qs)))
    variants = {"clean": [q for q, _, _ in qs],
                "typo_one_word": [typo(q, random.Random(i)) for i, (q, _, _) in enumerate(qs)],
                "typo_all_long_words": [typo_all(q, random.Random(i)) for i, (q, _, _) in enumerate(qs)],
                "keywords": [keywords(q) for q, _, _ in qs],
                "keywords_typo": [typo_all(keywords(q), random.Random(i)) for i, (q, _, _) in enumerate(qs)]}
    gold = [p for _, p, _ in qs]
    res = {"what": "fastmemory passage search vs MahaBodi lexical vs BM25 (same sample as bench_retrieval.py)",
           "provenance": provenance(), "fastmemory_version": getattr(fastmemory, "__version__", None),
           "data": "rajpurkar/squad validation, shuffle(seed=0)", "paragraphs": len(paras), "questions": len(qs),
           "question_ids": [i for _, _, i in qs], "systems": {}}
    pid = re.compile(r"(?:F_)?p_?(\d+)")

    def pages(ids):
        out = []
        for h in ids:
            m = pid.search(h)
            if m and int(m.group(1)) not in out:
                out.append(int(m.group(1)))
        return out[:5]

    # fastmemory passage search, plain text in, source "p<i>"
    t = time.time(); fm = fastmemory.SearchMemory([(p, "p%d" % i) for i, p in enumerate(paras)]); res["fastmemory_build_s"] = round(time.time() - t, 2)

    def fm_q(q):
        r = json.loads(fm.search(q, 5))
        return (pages([h["id"] for h in r["hits"]]) if r["matched"] else []), r["stage"], r["matched"], r["handoff"]
    score("fastmemory_search", fm_q, variants, gold, res)

    # MahaBodi lexical (the `bodi` arm of bench_retrieval.py)
    b = Bodi({}); b.ingest_batch([{"text": p, "format": "text", "source": "p%d" % i} for i, p in enumerate(paras)])

    def mb_q(q):
        r = b.query(q, k=5)
        return (pages([h["id"] for h in r["hits"]]) if r["matched"] else []), r["stage"], r["matched"], r["handoff"]
    score("mahabodi_lexical", mb_q, variants, gold, res)

    bm = BM25Okapi([toks(p) for p in paras])
    score("bm25", lambda q: ([int(x) for x in np.argsort(-bm.get_scores(toks(q)))[:5]], "bm25", True, False), variants, gold, res)

    for other in ("bm25", "mahabodi_lexical"):
        for vname in variants:
            a_ = [it["gold"] in it["top"] for it in res["systems"]["fastmemory_search"][vname]["items"]]
            b_ = [it["gold"] in it["top"] for it in res["systems"][other][vname]["items"]]
            res["systems"]["fastmemory_search"][vname]["mcnemar_recall5_vs_" + other] = mcnemar(a_, b_)
    out = a.out or os.path.join(ROOT, "research", "results", "retrieval_fastmemory_p%d.json" % a.paragraphs)
    json.dump(res, open(out, "w"), indent=1)
    print("DONE", out)


if __name__ == "__main__":
    main()
