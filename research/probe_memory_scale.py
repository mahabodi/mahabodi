"""Scale probe for MahaBodi's in-process memory (research/PREREG_MILLION_SCALE.md, "Scale path"): ingest the first N KILT
page abstracts (N = 10K, 30K, 100K by default; stop early if a step exceeds --max-seconds) and record ingest seconds, peak RSS,
density, and query latency for 50 fixed title queries. It reads page text only, no dev/test labels. It decides whether the
full ~5.9M-page pool can use in-process memory or needs the PostgreSQL path.

    .venv/bin/python research/probe_memory_scale.py --out research/results/probe_memory_scale.json
"""
import argparse, json, os, resource, sys, time
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from mahabodi import Bodi  # noqa: E402


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="/media/sda/data/kilt/pages/part-0000.parquet")
    ap.add_argument("--sizes", default="10000,30000,100000")
    ap.add_argument("--max-seconds", type=float, default=3600)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "probe_memory_scale.json"))
    a = ap.parse_args()
    t = pq.read_table(a.pages, columns=["title", "abstract"]).to_pydict()
    res = {"provenance": provenance(), "steps": []}
    for n in [int(x) for x in a.sizes.split(",")]:
        b = Bodi(); t0 = time.time()
        docs = [{"text": "# %s\n\n%s" % (ti, ab), "source": ti} for ti, ab in zip(t["title"][:n], t["abstract"][:n])]
        b.ingest_batch(docs)
        ing = time.time() - t0
        t1 = time.time(); dens = b.ensure_density(); dt = time.time() - t1
        qs = t["title"][:n:max(1, n // 50)][:50]
        t2 = time.time(); hits = sum(1 for q in qs if b.query(q, k=5).get("matched")); qt = (time.time() - t2) / len(qs)
        step = {"n": n, "ingest_s": round(ing, 1), "density_s": round(dt, 1), "query_ms": round(qt * 1000, 1),
                "title_queries_matched": "%d/%d" % (hits, len(qs)), "peak_rss_mb": round(rss_mb()),
                "density": {k: v for k, v in dens.items() if not isinstance(v, (list, dict))} if isinstance(dens, dict) else None}
        res["steps"].append(step); print(step, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        del b
        if ing + dt > a.max_seconds:
            res["stopped"] = "step %d took %.0fs > %.0fs" % (n, ing + dt, a.max_seconds); break
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
