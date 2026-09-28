"""End-to-end check of the entity-tag fix (README known issue): re-ingest every page from el_pg_affected_causes.json into
ONE Bodi memory with the current build (format "auto"), then check that each page with text has passages under its own
source, that no bare-name ATF is left, and that a query by the page title finds that page.

    KILT_DIR=... python research/check_tag_fix.py -> research/results/check_tag_fix.json
"""
import glob, json, os, re, sys
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from mahabodi import Bodi  # noqa: E402

K = os.environ.get("KILT_DIR", "/media/sda/data/kilt")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    pages = json.load(open(os.path.join(R, "el_pg_affected_causes.json")))["per_page"]
    bounds, n = [], 0
    for s in sorted(glob.glob(os.path.join(K, "pages", "part-*.parquet"))):
        m = pq.ParquetFile(s).metadata.num_rows
        bounds.append((s, n, n + m)); n += m
    docs = {}
    for p in pages:
        s, a, _ = next(x for x in bounds if x[1] <= p["row"] < x[2])
        t = pq.read_table(s, columns=["title", "abstract"]).slice(p["row"] - a, 1)
        docs[p["row"]] = (t.column("title")[0].as_py(), t.column("abstract")[0].as_py() or "")
    b = Bodi()
    b.ingest_batch([{"text": "# %s\n\n%s" % docs[r], "source": "pg%d" % r} for r in docs])
    snap = b.snapshot()
    ids = [x["id"] for x in snap["atfs"]]
    bare = [i for i in ids if not re.match(r"pg_\d+_", i)]
    per = []
    for p in pages:
        r = p["row"]; title = docs[r][0]
        own = [i for i in ids if i.startswith("pg_%d_" % r)]
        hits = [h["id"] for h in b.query(title, k=5).get("hits", [])]
        found = any(re.match(r"F_pg_%d_" % r, h) for h in hits)
        per.append({"row": r, "cause": p["cause"], "passages": len(own), "title_query_top5_finds_page": found})
    tagged = [x for x in per if x["cause"] == "entity_tags"]
    res = {"provenance": provenance(), "pages": len(per), "bare_name_atfs": bare,
           "tagged_pages_with_passages": sum(x["passages"] > 0 for x in tagged), "tagged_pages": len(tagged),
           "tagged_pages_found_by_title_top5": sum(x["title_query_top5_finds_page"] for x in tagged),
           "no_text_pages": [x for x in per if x["cause"] == "no_text"], "per_page": per}
    res["ok"] = not bare and res["tagged_pages_with_passages"] == len(tagged)
    json.dump(res, open(os.path.join(R, "check_tag_fix.json"), "w"), indent=1, ensure_ascii=False)
    print({k: v for k, v in res.items() if k not in ("per_page", "provenance")})


if __name__ == "__main__":
    main()
