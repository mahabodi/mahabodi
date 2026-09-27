"""Query latency of MahaBodi memory before/after a change, with identical results required. Run the same script under
two builds (two venvs) and compare the JSON files: same queries, same pages, top-10 hit ids per query recorded.

Pages: the first N KILT pages (as in probe_memory_scale.py). Queries: the held-out title/sentence/keyword queries from
research/results/bench_turbovec.json for the same N.

    <venv>/bin/python research/probe_query_speed.py --n 30000 --out research/results/probe_query_speed_<build>_<n>.json
"""
import argparse, hashlib, json, os, statistics, sys, time
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
import mahabodi  # noqa: E402
from mahabodi import Bodi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--queries", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = pq.read_table("/media/sda/data/kilt/pages/part-0000.parquet", columns=["title", "abstract"]).slice(0, a.n).to_pydict()
    Q = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "bench_turbovec.json")))["sizes"][str(a.n)]["queries"][:a.queries]
    b = Bodi()
    t0 = time.time()
    b.ingest_batch([{"text": "# %s\n\n%s" % (ti, ab), "source": "pg%d" % i} for i, (ti, ab) in enumerate(zip(t["title"], t["abstract"]))])
    ingest_s = time.time() - t0
    for x in Q[:20]:  # warm-up
        b.query(x["q"], k=10)
    lat, top = [], []
    for x in Q:
        s = time.perf_counter(); r = b.query(x["q"], k=10); lat.append((time.perf_counter() - s) * 1000)
        top.append([h["id"] for h in r.get("hits", [])])
    lat_s = sorted(lat)
    out = {"n": a.n, "queries": len(Q), "module": mahabodi.__file__, "ingest_s": round(ingest_s, 1),
           "latency_ms_p50": round(lat_s[len(lat_s) // 2], 2), "latency_ms_p95": round(lat_s[int(.95 * len(lat_s))], 2),
           "latency_ms_mean": round(statistics.mean(lat), 2),
           "by_type_p50": {ty: round(sorted(l for l, x in zip(lat, Q) if x["type"] == ty)[len([1 for x in Q if x["type"] == ty]) // 2], 2)
                           for ty in ("title", "sentence", "keywords")},
           "top10_sha256": hashlib.sha256(json.dumps(top).encode()).hexdigest(), "top10": top, "provenance": provenance()}
    json.dump(out, open(a.out, "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("top10", "provenance")})


if __name__ == "__main__":
    main()
