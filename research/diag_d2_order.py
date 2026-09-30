"""PREREG_SCALE_V2 diagnostic D2 (descriptive, DEV only): where store and in-process hit orders diverge.

500 dev mentions, the dev 100K pool, the existing devgate store namespace (current code on both sides).
  lexical:  in-process memory built WITHOUT an embedder (so `query` is the lexical cascade + spreading only) vs the
            store's `lexical` mode.
  hybrid:   in-process memory with the embedder (the benchmark query) vs the store's `hybrid` mode.
For each, top-60 hits per mention. At the first differing position the two ids are recorded with, on both sides:
score, level, degree (in process: 1-hop neighbours via traverse; store: node.degree), the store's dense similarity
(dense mode) and, for the non-Function neighbours of each id, their degrees (the spreading denominator sqrt(fdeg)).
The pair is classified by its scores on each side: exact tie, near-tie (< 1e-9) or a real difference.
Page level: items whose top-20 page SETS differ get the rank at which a page crossed the boundary and the scores of
the pages on either side of it.
Writes research/results/el_store_v2_diag_d2.json.

    KILT_DIR=... python research/diag_d2_order.py --dsn "..."
"""
import argparse, collections, gc, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from bench_el_store_v2 import EL, PAGE, R, ROOT, Pages, load_mentions, pages_of  # noqa: E402


def kind(s, x, y):
    if x not in s or y not in s:
        return "missing"
    d = abs(s[x] - s[y])
    return "tie" if d == 0 else "near_tie" if d < 1e-9 else "differs"


def page_scores(hits):
    """page row -> the score of its first (best) hit, in rank order."""
    out = {}
    for h in hits:
        m = PAGE.match(h["id"])
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = h["score"]
    return out


def compare(label, ms, inproc_q, store_q, deg_in, deg_st, nbrs_in, nbrs_st, dense_sim):
    kinds, first_pos, rows, set_rows, diffs = collections.Counter(), collections.Counter(), [], [], []
    stage_pairs = collections.Counter()
    for m in ms:
        q = m["mention"] or m["state"]
        ra, rb = inproc_q(q), store_q(q)
        ha, hb = ra.get("hits", []), rb.get("hits", [])
        stage_pairs["%s|%s" % (ra.get("stage"), rb.get("stage"))] += 1
        ia, ib = [h["id"] for h in ha], [h["id"] for h in hb]
        sa, sb = {h["id"]: h["score"] for h in ha}, {h["id"]: h["score"] for h in hb}
        diffs += [abs(sa[i] - sb[i]) for i in set(sa) & set(sb)]
        pa, pb = pages_of(ia), pages_of(ib)
        if set(pa) != set(pb):
            psa, psb = page_scores(ha), page_scores(hb)
            only_a, only_b = [p for p in pa if p not in pb], [p for p in pb if p not in pa]
            set_rows.append({"id": m["id"], "query": q[:60], "inproc_only": [(p, pa.index(p), psa.get(p), psb.get(p)) for p in only_a],
                             "store_only": [(p, pb.index(p), psb.get(p), psa.get(p)) for p in only_b],
                             "boundary_scores": {"inproc_20th": psa.get(pa[-1]) if pa else None, "store_20th": psb.get(pb[-1]) if pb else None}})
        if ia == ib:
            kinds["identical"] += 1
            continue
        k = next((j for j in range(min(len(ia), len(ib))) if ia[j] != ib[j]), min(len(ia), len(ib)))
        first_pos[k // 10 * 10] += 1
        if k >= min(len(ia), len(ib)):
            kinds["length_differs"] += 1
            continue
        x, y = ia[k], ib[k]
        kinds["inproc:%s store:%s" % (kind(sa, x, y), kind(sb, x, y))] += 1
        info = lambda i: {"inproc_score": sa.get(i), "store_score": sb.get(i), "inproc_degree": deg_in(i), "store_degree": deg_st(i),
                          "store_dense_sim": dense_sim(q).get(i) if dense_sim else None,
                          "inproc_nbr_degrees": nbrs_in(i), "store_nbr_degrees": nbrs_st(i)}
        rows.append({"id": m["id"], "query": q[:60], "pos": k, "stages": [ra.get("stage"), rb.get("stage")],
                     "inproc_first": x, "store_first": y, x: info(x), y: info(y)})
    sd = np.asarray(diffs)
    deg_mismatch = sum(1 for r in rows for i in (r["inproc_first"], r["store_first"]) if r[i]["inproc_degree"] != r[i]["store_degree"])
    nbr_mismatch = sum(1 for r in rows for i in (r["inproc_first"], r["store_first"]) if r[i]["inproc_nbr_degrees"] != r[i]["store_nbr_degrees"])
    out = {"n": len(ms), "first_divergence_kind": dict(kinds), "first_divergence_position": dict(sorted(first_pos.items())),
           "stage_pairs": dict(stage_pairs), "page_set_differs": len(set_rows),
           "degree_mismatches_at_divergence": deg_mismatch, "neighbour_degree_mismatches_at_divergence": nbr_mismatch,
           "per_id_abs_score_diff": {"n": int(sd.size), "max": float(sd.max()) if sd.size else None,
                                     "p99": float(np.percentile(sd, 99)) if sd.size else None,
                                     "share_exactly_equal": float(np.mean(sd == 0)) if sd.size else None},
           "divergences": rows, "page_set_items": set_rows}
    print(label, json.dumps({k: v for k, v in out.items() if k not in ("divergences", "page_set_items")}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--namespace", default="devgate")
    a = ap.parse_args()
    import psycopg
    from mahabodi import Bodi
    ms = load_mentions("dev")
    pool = np.load(os.path.join(EL, "pool_dev_100000_mq.npy")).tolist()
    P = Pages()
    ab = P.abstracts(pool)
    docs = [{"text": "# %s\n\n%s" % (P.titles[r], ab[r]), "source": "pg%d" % r} for r in pool]
    db = psycopg.connect(a.dsn, autocommit=True)

    def deg_st(i):
        r = db.execute("SELECT degree FROM mahabodi_store.node WHERE ns = %s AND id = %s", (a.namespace, i)).fetchone()
        return r[0] if r else None

    def nbrs_st(i):
        # each non-Function neighbour with its count of Function neighbours: the spreading denominator sqrt(fdeg)
        nb = [r[0] for r in db.execute("""SELECT DISTINCT n.id FROM mahabodi_store.link l JOIN mahabodi_store.node n ON n.ns = l.ns AND n.id = CASE WHEN l.a = %s THEN l.b ELSE l.a END
                             WHERE l.ns = %s AND (l.a = %s OR l.b = %s) AND n.level <> 0""", (i, a.namespace, i, i)).fetchall()]
        out = []
        for j in nb:
            f = db.execute("""SELECT count(DISTINCT n.id) FROM mahabodi_store.link l JOIN mahabodi_store.node n ON n.ns = l.ns AND n.id = CASE WHEN l.a = %s THEN l.b ELSE l.a END
                              WHERE l.ns = %s AND (l.a = %s OR l.b = %s) AND n.level = 0 AND n.id <> %s""", (j, a.namespace, j, j, j)).fetchone()[0]
            out.append((j, int(f)))
        return sorted(out)

    def inproc_fns(b):
        # traverse's depth-1 nodes = adj[i] (adj is deduplicated, no self-loops: graph.rs), i.e. Graph::degree
        def one_hop(i):
            t = b.call("traverse", start=i, hops=1, limit=10_000_000)
            return [n for n in t.get("nodes", []) if n["depth"] == 1] if t.get("start") == i else None

        def deg(i):
            h = one_hop(i)
            return None if h is None else len(h)

        def nbrs(i):
            h = one_hop(i) or []
            return sorted((n["id"], len([x for x in one_hop(n["id"]) or [] if x["level"] == "function"])) for n in h if n["level"] != "function")
        return deg, nbrs

    store = Bodi(); store.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    store.call("store_open", dsn=a.dsn, namespace=a.namespace, create=False, vector_type="vector")
    dense_cache = {}

    def dense_sim(q):
        if q not in dense_cache:
            dense_cache[q] = {h["id"]: h["score"] for h in store.call("store_query", q=q, k=60, mode="dense").get("hits", [])}
        return dense_cache[q]

    out = {"diagnostic": "PREREG_SCALE_V2 D2 (descriptive, dev only)", "provenance": provenance(), "namespace": a.namespace,
           "mention_ids": [m["id"] for m in ms], "link_columns_checked": db.execute(
               "SELECT string_agg(column_name, ',' ORDER BY ordinal_position) FROM information_schema.columns WHERE table_schema = 'mahabodi_store' AND table_name = 'link'").fetchone()[0]}

    # lexical: in-process without an embedder vs store lexical mode
    lx = Bodi(); lx.ingest_batch(docs)
    out["graph_without_embedder"] = {"stats": lx.call("stats"), "density": lx.call("density")}
    d_in, n_in = inproc_fns(lx)
    out["lexical"] = compare("LEXICAL", ms, lambda q: lx.query(q, k=60), lambda q: store.call("store_query", q=q, k=60, mode="lexical"),
                             d_in, deg_st, n_in, nbrs_st, None)
    del lx; gc.collect()

    # hybrid: in-process with the embedder vs store hybrid mode
    hy = Bodi(); hy.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8); hy.ingest_batch(docs)
    out["graph_with_embedder"] = {"stats": hy.call("stats"), "density": hy.call("density")}
    print("GRAPHS", json.dumps({"without": out["graph_without_embedder"]["stats"], "with": out["graph_with_embedder"]["stats"]})[:600], flush=True)
    d_in, n_in = inproc_fns(hy)
    out["hybrid"] = compare("HYBRID", ms, lambda q: hy.query(q, k=60), lambda q: store.call("store_query", q=q, k=60),
                            d_in, deg_st, n_in, nbrs_st, dense_sim)
    json.dump(out, open(os.path.join(R, "el_store_v2_diag_d2.json"), "w"), indent=1, default=str)
    print("D2-DONE", flush=True)


if __name__ == "__main__":
    main()
