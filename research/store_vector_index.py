"""PREREG_SCALE_V2 step 2 at dev 100K (clarification 3a), on the dev-gate store namespace, dev queries only:
  1. halfvec check: exact top-50 with float4 vectors vs the same vectors cast to halfvec (fp16); mean overlap@50.
     halfvec is used at 5.9M if the overlap is >= 0.99.
  2. IVFFlat, lists = ceil(sqrt(rows)): recall@50 vs exact and latency for probes {10, 20, 40, 80, 160, 320}. The
     smallest probes with recall@50 >= 0.98 is the setting for the test-100K bridge (else 320, flagged).
  3. HNSW (m 16, ef_construction 64), descriptive only: build time, recall@50 and latency for ef_search {40, 100, 200, 400}.
Query vectors come from the MahaBodi embedder (the same one the store uses).

    python research/store_vector_index.py --dsn "..." --namespace devgate
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

EL = os.path.join(os.environ.get("KILT_DIR", "/media/sda/data/kilt"), "el")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lit(v):
    return "[" + ",".join("%.7f" % x for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--namespace", default="devgate")
    ap.add_argument("--mwm", default="10GB")
    a = ap.parse_args()
    import psycopg
    from mahabodi import Bodi
    M = [json.loads(l) for l in open(os.path.join(EL, "mentions_mq.jsonl"))]
    ms = [m for m in M if m["split"] == "dev"]
    b = Bodi(); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    qs = b.embed_text([m["mention"] or m["state"] for m in ms])
    c = psycopg.connect(a.dsn, autocommit=True)
    ns = a.namespace
    rows = c.execute("SELECT count(*) FROM mahabodi_store.vec WHERE ns = %s", (ns,)).fetchone()[0]
    c.execute("DROP INDEX IF EXISTS mahabodi_store.vec_hnsw"); c.execute("DROP INDEX IF EXISTS mahabodi_store.vec_ivfflat")

    def top(q, cast="", n=50):
        return [r[0] for r in c.execute("SELECT node_id FROM mahabodi_store.vec WHERE ns = %%s ORDER BY embedding%s <#> %%s::text::%s, node_id LIMIT %d"
                                        % (cast, "halfvec" if cast else "vector", n), (ns, lit(q))).fetchall()]

    c.execute("SET enable_indexscan = off"); c.execute("SET enable_bitmapscan = off")
    t = time.time(); exact = [top(q) for q in qs]; exact_s = time.time() - t
    half = [top(q, "::halfvec") for q in qs]
    overlap = float(np.mean([len(set(x) & set(y)) / 50 for x, y in zip(exact, half)]))
    print("halfvec overlap@50", round(overlap, 4), flush=True)
    c.execute("RESET enable_indexscan"); c.execute("RESET enable_bitmapscan")
    out = {"step": "PREREG_SCALE_V2 step 2 at dev 100K (clarification 3a)", "provenance": provenance(), "namespace": ns, "rows": rows,
           "queries": len(qs), "mention_ids": [m["id"] for m in ms], "exact_scan_s_total": round(exact_s, 1),
           "halfvec_overlap_at_50": round(overlap, 4), "halfvec_use_at_5_9M": overlap >= 0.99, "maintenance_work_mem": a.mwm}

    def sweep(setting, values):
        res = {}
        for v in values:
            c.execute("SET %s = %d" % (setting, v))
            lat, rec = [], []
            for q, ex in zip(qs, exact):
                t = time.perf_counter(); got = top(q); lat.append((time.perf_counter() - t) * 1000)
                rec.append(len(set(got) & set(ex)) / 50)
            res[str(v)] = {"recall_at_50": round(float(np.mean(rec)), 4), "latency_ms_p50": round(float(np.median(lat)), 2),
                           "latency_ms_p95": round(float(np.percentile(lat, 95)), 2)}
            print(setting, v, res[str(v)], flush=True)
        return res

    lists = math.ceil(math.sqrt(rows))
    c.execute("SET maintenance_work_mem = '%s'" % a.mwm); c.execute("SET max_parallel_maintenance_workers = 6")
    t = time.time(); c.execute("CREATE INDEX vec_ivfflat ON mahabodi_store.vec USING ivfflat (embedding vector_ip_ops) WITH (lists = %d)" % lists)
    ivf_build = time.time() - t
    ivf = sweep("ivfflat.probes", [10, 20, 40, 80, 160, 320])
    ok = [int(p) for p, r in ivf.items() if r["recall_at_50"] >= 0.98]
    out["ivfflat"] = {"lists": lists, "build_s": round(ivf_build, 1), "sweep": ivf,
                      "selected_probes_for_bridge": min(ok) if ok else 320, "reached_0_98": bool(ok)}
    c.execute("DROP INDEX mahabodi_store.vec_ivfflat")
    t = time.time(); c.execute("CREATE INDEX vec_hnsw ON mahabodi_store.vec USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 64)")
    hnsw_build = time.time() - t
    out["hnsw_descriptive"] = {"build_s": round(hnsw_build, 1), "sweep": sweep("hnsw.ef_search", [40, 100, 200, 400])}
    c.execute("DROP INDEX mahabodi_store.vec_hnsw")
    json.dump(out, open(os.path.join(R, "store_vector_index_dev100k.json"), "w"), indent=1)
    print("STEP2-DONE", json.dumps({k: out[k] for k in ("halfvec_overlap_at_50", "halfvec_use_at_5_9M")}), out["ivfflat"]["selected_probes_for_bridge"])


if __name__ == "__main__":
    main()
