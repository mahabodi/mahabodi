"""PREREG_SCALE_V2 diagnostic D1 (descriptive only): why the bridge's exact store differs from in-process M.

  (a) in-process M at the current code on the test 100K pool (as bench_el: ingest_batch, query k = 60 -> 20 pages ->
      decide n = 48), per-item shortlists and predictions;
  (b) store exact search with float4 vectors on namespace t100kf, per-item shortlists and predictions;
  (c) pairwise: prediction agreement, exact McNemar, identical shortlists (order), equal shortlist sets.
Decision cache off everywhere (clarification 3e). Writes research/results/el_store_v2_diag_d1.json.

    KILT_DIR=... python research/diag_bridge_d1.py --dsn "..."
"""
import argparse, gc, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from bench_el_store_v2 import (EL, R, ROOT, Pages, laya_choice, load_mentions, load_pool, mcnemar, no_decision_cache,  # noqa: E402
                               pages_of, run_store_arm, wilson)


def pair(name_a, a, name_b, b, gold):
    ca = [p == g for p, g in zip(a["pred"], gold)]; cb = [p == g for p, g in zip(b["pred"], gold)]
    d = {"a": name_a, "b": name_b, "accuracy_a": round(float(np.mean(ca)), 4), "accuracy_b": round(float(np.mean(cb)), 4),
         "pred_agree": int(sum(x == y for x, y in zip(a["pred"], b["pred"]))), "mcnemar": mcnemar(ca, cb)}
    if a.get("shortlists") and b.get("shortlists"):
        d["identical_shortlists"] = int(sum(x == y for x, y in zip(a["shortlists"], b["shortlists"])))
        d["equal_shortlist_sets"] = int(sum(set(x) == set(y) for x, y in zip(a["shortlists"], b["shortlists"])))
        same = [i for i, (x, y) in enumerate(zip(a["shortlists"], b["shortlists"])) if x == y]
        d["pred_agree_where_shortlists_identical"] = int(sum(a["pred"][i] == b["pred"][i] for i in same))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--laya", default=os.path.join(ROOT, "models", "laya-v2"))
    a = ap.parse_args()
    from mahabodi import Bodi
    ms = load_mentions("test"); gold = [m["gold_row"] for m in ms]
    el = json.load(open(os.path.join(R, "bench_el.json")))
    assert el["mention_ids"] == [m["id"] for m in ms]
    bridge = json.load(open(os.path.join(R, "el_store_v2_bridge.json")))
    assert bridge["mention_ids"] == [m["id"] for m in ms]
    pool = np.load(os.path.join(EL, "pool_test_100000_mq.npy")).tolist()
    P = Pages()
    out = {"diagnostic": "PREREG_SCALE_V2 D1 (descriptive only)", "provenance": provenance(), "decide_cache": False,
           "mention_ids": [m["id"] for m in ms], "gold_rows": gold, "times": {}}

    # (a) in-process M at the current code
    ck_a = os.path.join(R, "el_store_v2_diag_d1_inproc.ckpt.json")
    if os.path.exists(ck_a):
        inproc = json.load(open(ck_a))
    else:
        b = no_decision_cache(Bodi()); b.load_laya(a.laya, intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
        t = time.time()
        ab = P.abstracts(pool)
        b.ingest_batch([{"text": "# %s\n\n%s" % (P.titles[r], ab[r]), "source": "pg%d" % r} for r in pool])
        out["times"]["inproc_build_s"] = round(time.time() - t, 1)
        sh_all, preds = [], []
        for i, m in enumerate(ms):
            sh = pages_of([h["id"] for h in b.query(m["mention"] or m["state"], k=60).get("hits", [])])
            sh_all.append(sh); preds.append(laya_choice(b, m["state"], sh, P, 48, decide=True))
            if i % 100 == 0:
                print("inproc", i, "/", len(ms), flush=True)
        inproc = {"pred": preds, "shortlists": sh_all, "shortlist_hit": [g in s for g, s in zip(gold, sh_all)]}
        json.dump(inproc, open(ck_a, "w"))
        del b; gc.collect()
    c = [p == g for p, g in zip(inproc["pred"], gold)]
    inproc.update({"accuracy": round(float(np.mean(c)), 4), "ci95": wilson(sum(c), len(c)), "shortlist_recall": round(float(np.mean(inproc["shortlist_hit"])), 4)})

    # (b) store exact, float4, namespace t100kf
    b = no_decision_cache(Bodi()); b.load_laya(a.laya, intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    b.call("store_open", dsn=a.dsn, namespace="t100kf", vector_type="vector")
    out["times"]["store_f4_load_s"] = load_pool(b, pool, P, os.path.join(R, "el_store_v2_diag_d1_t100kf.load.json"))
    t = time.time(); b.call("store_build_index"); out["times"]["store_f4_build_s"] = round(time.time() - t, 1)
    t = time.time(); out["times"]["store_f4_density"] = b.call("store_ensure_density"); out["times"]["store_f4_density_s"] = round(time.time() - t, 1)
    out["store_f4_stats"] = b.call("store_stats")
    store_f4 = run_store_arm(b, ms, P, os.path.join(R, "el_store_v2_diag_d1_f4.ckpt.json"))

    old = {"pred": el["stages"]["100000"]["arms"]["M"]["pred"]}
    half = bridge["store_exact"]
    out["arms"] = {"inproc_current": {k: v for k, v in inproc.items()}, "store_exact_float4": store_f4}
    out["pairs"] = [pair("inproc_bench_el (old code)", old, "inproc_current", inproc, gold),
                    pair("inproc_current", inproc, "store_exact_halfvec (bridge)", half, gold),
                    pair("inproc_current", inproc, "store_exact_float4", store_f4, gold),
                    pair("store_exact_float4", store_f4, "store_exact_halfvec (bridge)", half, gold)]
    json.dump(out, open(os.path.join(R, "el_store_v2_diag_d1.json"), "w"), indent=1, default=str)
    for p_ in out["pairs"]:
        print("PAIR", json.dumps(p_), flush=True)
    print("D1-DONE")


if __name__ == "__main__":
    main()
