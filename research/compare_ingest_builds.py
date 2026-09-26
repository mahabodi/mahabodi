"""Fingerprint the memory one MahaBodi build makes from the first N KILT pages, so two builds (before and after
the parallel-ingest change, 4aa4e6e) can be compared item for item. Run the same script under each build's Python and diff
the JSON files.

Records per N: the ingest_batch report (ATFs, nodes, edges, blocks, and the density outcome of the automatic density
pass), the separate density() report, and the top-1 hit id for 200 fixed title queries (the list and its sha256).

    <venv>/bin/python research/compare_ingest_builds.py --sizes 10000,30000 --out research/results/compare_ingest_<build>.json
"""
import argparse, hashlib, json, os, sys
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
import mahabodi  # noqa: E402
from mahabodi import Bodi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="/media/sda/data/kilt/pages/part-0000.parquet")
    ap.add_argument("--sizes", default="10000,30000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    t = pq.read_table(a.pages, columns=["title", "abstract"]).slice(0, max(sizes)).to_pydict()
    res = {"mahabodi_module": mahabodi.__file__, "provenance": provenance(), "runs": []}
    for n in sizes:
        b = Bodi()
        rep = b.ingest_batch([{"text": "# %s\n\n%s" % (ti, ab), "source": ti} for ti, ab in zip(t["title"][:n], t["abstract"][:n])])
        top1 = []
        for q in t["title"][:200]:
            h = b.query(q, k=5).get("hits") or []
            top1.append(h[0]["id"] if h else None)
        res["runs"].append({"n": n, "ingest_report": rep, "density_report": b.density(), "top1": top1,
                            "top1_sha256": hashlib.sha256(json.dumps(top1).encode()).hexdigest()})
        print(n, json.dumps(rep)[:300], res["runs"][-1]["top1_sha256"][:16], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        del b


if __name__ == "__main__":
    main()
