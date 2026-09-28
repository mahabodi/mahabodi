"""Why each page in el_pg_affected_pages.json has no passage in the PG store: fastmemory entity tags in its text (the
known parser issue) or no text at all. Reads page text only.

    KILT_DIR=... .venv/bin/python research/el_affected_causes.py -> research/results/el_pg_affected_causes.json
"""
import glob, json, os, re, sys
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

K = os.environ.get("KILT_DIR", "/media/sda/data/kilt")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TAG = re.compile(r"\((Component|Block|Function|Data|Access|Event)\s+([A-Za-z0-9_]+)\)")  # fastmemory's parser regex


def main():
    rows = json.load(open(os.path.join(R, "el_pg_affected_pages.json")))["affected_rows"]
    bounds, n = [], 0
    for s in sorted(glob.glob(os.path.join(K, "pages", "part-*.parquet"))):
        m = pq.ParquetFile(s).metadata.num_rows
        bounds.append((s, n, n + m)); n += m
    per = []
    for r in rows:
        s, a, _ = next(x for x in bounds if x[1] <= r < x[2])
        t = pq.read_table(s, columns=["title", "abstract"]).slice(r - a, 1)
        ti, ab = t.column("title")[0].as_py(), t.column("abstract")[0].as_py() or ""
        tags = TAG.findall(ab)
        cause = "entity_tags" if tags else ("no_text" if not any(c.isalnum() for c in ab) else "other")
        per.append({"row": r, "title": ti, "abstract_chars": len(ab), "tags": len(tags), "cause": cause})
    out = {"provenance": provenance(), "pages": len(per), "per_page": per,
           "by_cause": {c: sum(p["cause"] == c for p in per) for c in ("entity_tags", "no_text", "other")}}
    json.dump(out, open(os.path.join(R, "el_pg_affected_causes.json"), "w"), indent=1, ensure_ascii=False)
    print(out["by_cause"])


if __name__ == "__main__":
    main()
