"""Held-out retrieval benchmark: do Bodi queries find the right memory for questions written
independently of the memory's wording?

Data: SQuAD v1.1 validation (rajpurkar/squad). Crowdworkers wrote the questions while
reading the paragraph, so queries paraphrase rather than copy it. Sample: first P unique
paragraphs of the seeded shuffle, then Q seeded questions about them.

Queries: `clean` (question as written) and `typo` (one word of >= 5 letters loses one
character, seeded).

Failure is defined up front: the question's paragraph is not in the top-k. A `hub`
fallback result counts as a miss. Reported per system:
  recall@1, recall@5, handoff rate, confidently-wrong rate (matched, no handoff, and the
  paragraph is not in the top-5), answered-correctly-without-handoff.
  Bodi's index is BM25-based: comparable recall to bm25 is expected; its added value is
  the handoff / confidently-wrong behaviour and typo tolerance, reported separately.

Baselines:
  bm25          rank_bm25.BM25Okapi over the same paragraphs (lower-cased alphanumeric tokens)
  fastmemory    fastmemory's PyPI package (process_markdown + its substring search rule), run
                under sandbox-exec with outbound network denied (the package posts telemetry).
                Its NLTK extractor keeps neither passage text nor ids, so only "returned
                anything" can be scored for it.

    .venv/bin/python research/bench_retrieval.py --paragraphs 300 --questions 600
"""
import argparse, json, os, random, re, subprocess, sys, tempfile, time
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from mahabodi import Bodi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def typo(q, rng):
    words = [w for w in re.findall(r"[A-Za-z]+", q) if len(w) >= 5]
    if not words:
        return q
    w = rng.choice(words)
    i = rng.randrange(1, len(w) - 1)
    return q.replace(w, w[:i] + w[i + 1:], 1)


def typo_all(q, rng):
    """Every word of >= 5 letters loses one interior character."""
    def f(m):
        w = m.group(0)
        if len(w) < 5:
            return w
        i = rng.randrange(1, len(w) - 1)
        return w[:i] + w[i + 1:]
    return re.sub(r"[A-Za-z]+", f, q)


STOP = set("the a an of in on to for is was what which who whom when where why how did does do are were by with from as at that this".split())


def keywords(q, n=2):
    """The n longest content words (a terse agent-style query)."""
    ws = sorted({w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP}, key=lambda w: (-len(w), w))
    return " ".join(ws[:n])


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


FM_SCRIPT = r'''
import json, sys, fastmemory
paras, queries = json.load(open(sys.argv[1]))
text = "\n\n".join(paras)
topo = json.loads(fastmemory.process_markdown(text))
def has(v, q):
    q = q.lower()
    if isinstance(v, dict):
        for k in ("name", "action", "id"):
            if isinstance(v.get(k), str) and q in v[k].lower(): return True
        return any(has(n, q) for n in v.get("nodes", []) + v.get("sub_blocks", []))
    return False
out = {"blocks": len(topo), "any": [sum(1 for b in topo if has(b, q)) > 0 for q in queries["full"]],
       "any_keyword": [sum(1 for b in topo if has(b, q)) > 0 for q in queries["keyword"]]}
json.dump(out, sys.stdout)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paragraphs", type=int, default=300)
    ap.add_argument("--questions", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "retrieval.json"))
    a = ap.parse_args()
    d = load_dataset("rajpurkar/squad", split="validation").shuffle(seed=0)
    paras, pidx, qs = [], {}, []
    for r in d:
        c = r["context"]
        if c not in pidx:
            if len(paras) >= a.paragraphs:
                continue
            pidx[c] = len(paras)
            paras.append(c)
        qs.append((r["question"], pidx[c], r["id"]))
    rng = random.Random(0)
    qs = rng.sample(qs, min(a.questions, len(qs)))
    variants = {
        "clean": [q for q, _, _ in qs],
        "typo_one_word": [typo(q, random.Random(i)) for i, (q, _, _) in enumerate(qs)],
        "typo_all_long_words": [typo_all(q, random.Random(i)) for i, (q, _, _) in enumerate(qs)],
        "keywords": [keywords(q) for q, _, _ in qs],
        "keywords_typo": [typo_all(keywords(q), random.Random(i)) for i, (q, _, _) in enumerate(qs)],
    }
    gold = [p for _, p, _ in qs]
    res = {"data": "rajpurkar/squad validation, shuffle(seed=0)", "paragraphs": len(paras), "questions": len(qs),
           "question_ids": [i for _, _, i in qs], "systems": {}}

    # --- Bodi (lexical cascade) and Bodi hybrid (+ dense MiniLM, RRF) -----------------------
    hyb_cfg = os.path.join(ROOT, "research", "results", "tune_hybrid.json")
    systems = [("bodi", {})]
    if os.path.exists(os.path.join(ROOT, "models", "minilm", "model.onnx")) and os.path.exists(hyb_cfg):
        systems.append(("bodi_hybrid", {"embedder_dir": os.path.join(ROOT, "models", "minilm"),
                                        "dense_min_similarity": json.load(open(hyb_cfg))["best"]}))
    fc = os.path.join(ROOT, "research", "results", "tune_hybrid_fuzzycheck.json")
    if len(systems) > 1 and os.path.exists(fc):
        # "safety mode", fixed in advance from SQuAD train: the threshold with the most correct
        # answers among those keeping keywords_typo confidently-wrong < 0.10 on train
        t = json.load(open(fc))
        ok = [(float(k), v) for k, v in t.items() if isinstance(v, dict) and v["keywords_typo"]["confidently_wrong"] < 0.10]
        if ok:
            thr = max(ok, key=lambda kv: (kv[1]["keywords_typo"]["answered_correctly_without_handoff"], -kv[0]))[0]
            systems.append(("bodi_hybrid_safe", {"embedder_dir": os.path.join(ROOT, "models", "minilm"),
                                                 "dense_min_similarity": thr, "fuzzy_requires_dense": True}))
    for sysname, cfg in systems:
        run_bodi(sysname, cfg, paras, variants, gold, res)
    res["systems"]["bodi_hybrid_config"] = systems[1][1] if len(systems) > 1 else None
    res["systems"]["bodi_hybrid_safe_config"] = systems[2][1] if len(systems) > 2 else None
    _rest(res, paras, variants, gold, a)


def run_bodi(sysname, cfg, paras, variants, gold, res):
    b = Bodi(cfg)
    t = time.time()
    b.ingest_batch([{"text": p, "format": "text", "source": "p%d" % i} for i, p in enumerate(paras)])
    ingest_s = time.time() - t
    dens = b.density()
    res[sysname + "_memory"] = {"ingest_seconds_total": round(ingest_s, 2), "auto_density": True, "atfs": b.stats()["atfs"],
                          "density_passes": dens["passes"], "probe_recall": dens["probe_recall"]}
    def para_of(hid):
        m = re.match(r"F_p_(\d+)_", hid)
        assert m, "unparseable hit id %r" % hid
        return int(m.group(1))
    for vname, qlist in variants.items():
        r1 = r5 = hand = wrong = answered = over = 0
        stages, lat, items = {}, [], []
        for q, g in zip(qlist, gold):
            t = time.perf_counter(); r = b.query(q, k=5); lat.append((time.perf_counter() - t) * 1000)
            ps = [para_of(h["id"]) for h in r["hits"]] if r["matched"] else []
            r1 += bool(ps and ps[0] == g); r5 += g in ps
            hand += r["handoff"]; wrong += (r["matched"] and not r["handoff"] and g not in ps)
            answered += (r["matched"] and not r["handoff"] and g in ps)
            over += (r["handoff"] and g in ps)
            stages[r["stage"]] = stages.get(r["stage"], 0) + 1
            items.append({"q": q, "gold": g, "top": ps, "stage": r["stage"], "handoff": r["handoff"]})
        n = len(qlist)
        res["systems"].setdefault(sysname, {})[vname] = {
            "recall@1": round(r1 / n, 4), "recall@5": round(r5 / n, 4), "handoff_rate": round(hand / n, 4),
            "confidently_wrong_rate": round(wrong / n, 4),
            "answered_correctly_without_handoff": round(answered / n, 4),
            "handed_off_although_correct": round(over / n, 4), "stages": stages,
            "query_ms_p50": round(float(np.percentile(lat, 50)), 3), "query_ms_p95": round(float(np.percentile(lat, 95)), 3), "items": items}
        print(sysname, vname, {k: v for k, v in res["systems"][sysname][vname].items() if k != "items"}, flush=True)


def _rest(res, paras, variants, gold, a):
    # --- BM25 ---------------------------------------------------------------------
    bm = BM25Okapi([toks(p) for p in paras])
    for vname, qlist in variants.items():
        r1 = r5 = 0
        tops = []
        for q, g in zip(qlist, gold):
            top = [int(x) for x in np.argsort(-bm.get_scores(toks(q)))[:5]]
            tops.append(top)
            r1 += top[0] == g; r5 += g in top
        res["systems"].setdefault("bm25", {})[vname] = {"recall@1": round(r1 / len(qlist), 4), "recall@5": round(r5 / len(qlist), 4), "top5": tops}
        print("bm25", vname, {k: v for k, v in res["systems"]["bm25"][vname].items() if k != "top5"}, flush=True)
    # paired recall@5 vs BM25, same queries (exact two-sided McNemar)
    from bench import mcnemar
    for sysname in [k for k in res["systems"] if k.startswith("bodi") and not k.endswith("_config")]:
        for vname in variants:
            bo = [it["gold"] in it["top"] for it in res["systems"][sysname][vname]["items"]]
            bb = [g in t for g, t in zip(gold, res["systems"]["bm25"][vname]["top5"])]
            res["systems"][sysname][vname]["mcnemar_recall5_vs_bm25"] = mcnemar(bo, bb)

    # --- fastmemory (PyPI), network denied ---------------------------------------------
    kw = [keywords(q, 1) for q in variants["clean"]]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([paras, {"full": variants["clean"], "keyword": kw}], f)
        inp = f.name
    script = os.path.join(tempfile.gettempdir(), "fm_baseline.py")
    open(script, "w").write(FM_SCRIPT)
    prof = "(version 1)(allow default)(deny network-outbound (remote ip))"
    t = time.time()
    p = subprocess.run(["sandbox-exec", "-p", prof, sys.executable, script, inp], capture_output=True, text=True,
                       env={**os.environ, "NLTK_DATA": os.path.expanduser("~/nltk_data")})
    el = time.time() - t
    if p.returncode == 0:
        fm = json.loads(p.stdout)
        res["systems"]["fastmemory_pypi"] = {"note": "no passage ids/text in its topology: only 'returned any block' is scorable",
                                             "blocks": fm["blocks"], "build_seconds": round(el, 1),
                                             "returned_anything_full_question": round(float(np.mean(fm["any"])), 4),
                                             "returned_anything_keyword": round(float(np.mean(fm["any_keyword"])), 4)}
    else:
        res["systems"]["fastmemory_pypi"] = {"error": p.stderr[-800:]}
    print("fastmemory", res["systems"]["fastmemory_pypi"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
