"""PREREG_SCALE_V2 diagnostic D2 (descriptive, DEV only): where store and in-process hit orders diverge.

For the 500 dev mentions, the dev 100K pool in process (current code) and the existing devgate store namespace each
return their top-60 hits (hybrid, the benchmark query). At the first position where the two id lists differ, the pair
of ids is classified by their scores on each side:
  tie_broken_differently  the two ids have exactly equal scores on a side (so the tie-break decided the order)
  near_tie                |score difference| < 1e-9 on a side, not exactly equal
  score_differs           otherwise (the sides score these hits differently)
Per-id score differences between the sides (for ids in both lists) are summarised too.
Writes research/results/el_store_v2_diag_d2.json.

    KILT_DIR=... python research/diag_d2_order.py --dsn "..."
"""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from bench_el_store_v2 import EL, R, ROOT, Pages, load_mentions  # noqa: E402


def classify(sa, sb, x, y):
    """x, y: the ids at the first divergence (x first on side a, y first on side b)."""
    out = []
    for s in (sa, sb):
        if x in s and y in s:
            d = abs(s[x] - s[y])
            out.append("tie" if d == 0 else "near_tie" if d < 1e-9 else "differs")
        else:
            out.append("missing")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--namespace", default="devgate")
    a = ap.parse_args()
    from mahabodi import Bodi
    ms = load_mentions("dev")
    pool = np.load(os.path.join(EL, "pool_dev_100000_mq.npy")).tolist()
    P = Pages()
    b = Bodi(); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    ab = P.abstracts(pool)
    b.ingest_batch([{"text": "# %s\n\n%s" % (P.titles[r], ab[r]), "source": "pg%d" % r} for r in pool])
    b.call("store_open", dsn=a.dsn, namespace=a.namespace, create=False, vector_type="vector")
    kinds, stage_pairs, first_pos, rows = collections.Counter(), collections.Counter(), collections.Counter(), []
    score_diffs = []
    for m in ms:
        q = m["mention"] or m["state"]
        ra = b.query(q, k=60); rb = b.call("store_query", q=q, k=60)
        ia = [h["id"] for h in ra.get("hits", [])]; ib = [h["id"] for h in rb.get("hits", [])]
        sa = {h["id"]: h["score"] for h in ra.get("hits", [])}; sb = {h["id"]: h["score"] for h in rb.get("hits", [])}
        stage_pairs[(ra.get("stage"), rb.get("stage"))] += 1
        score_diffs += [abs(sa[i] - sb[i]) for i in set(sa) & set(sb)]
        if ia == ib:
            kinds["identical"] += 1
            continue
        k = next((j for j in range(min(len(ia), len(ib))) if ia[j] != ib[j]), min(len(ia), len(ib)))
        first_pos[k // 10 * 10] += 1
        if k >= min(len(ia), len(ib)):
            kinds["length_differs"] += 1
            continue
        x, y = ia[k], ib[k]
        c = classify(sa, sb, x, y)
        kinds["a:%s b:%s" % tuple(c)] += 1
        rows.append({"id": m["id"], "query": q[:60], "pos": k, "inproc_first": x, "store_first": y,
                     "inproc_scores": [sa.get(x), sa.get(y)], "store_scores": [sb.get(x), sb.get(y)],
                     "stage": [ra.get("stage"), rb.get("stage")]})
    sd = np.asarray(score_diffs)
    out = {"diagnostic": "PREREG_SCALE_V2 D2 (descriptive, dev only)", "provenance": provenance(), "namespace": a.namespace,
           "n": len(ms), "first_divergence_kind": dict(kinds), "first_divergence_position": dict(sorted(first_pos.items())),
           "stage_pairs": {"%s|%s" % k: v for k, v in stage_pairs.items()},
           "per_id_abs_score_diff": {"n": int(sd.size), "max": float(sd.max()) if sd.size else None,
                                     "p99": float(np.percentile(sd, 99)) if sd.size else None,
                                     "share_exactly_equal": float(np.mean(sd == 0)) if sd.size else None},
           "examples": rows[:60]}
    json.dump(out, open(os.path.join(R, "el_store_v2_diag_d2.json"), "w"), indent=1, default=str)
    print("D2", json.dumps({k: out[k] for k in ("first_divergence_kind", "per_id_abs_score_diff", "stage_pairs")}), flush=True)


if __name__ == "__main__":
    main()
