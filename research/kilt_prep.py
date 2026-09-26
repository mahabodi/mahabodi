"""Build the KILT page table for the million-scale benchmarks (research/PREREG_MILLION_SCALE.md).

Streams kilt_knowledgesource.json (one Wikipedia page per line, ~5.9M pages) and writes Parquet shards:
  pages/part-XXXX.parquet: wikipedia_id (str), title (str), abstract (str: the first non-empty
  paragraphs after the title line, up to ABSTRACT_CHARS characters), n_paragraphs (int),
  paragraphs (list[str]: every paragraph except "Section::::" markers, for FEVER evidence).
Nothing is filtered or deduplicated; every page in the dump becomes one row. A manifest with row counts, the
dump's sha256 (from SHA256SUMS) and the script's own provenance is written next to the shards.

    .venv/bin/python research/kilt_prep.py --kilt /media/sda/data/kilt --out /media/sda/data/kilt/pages
"""
import argparse, json, os, sys, time
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

ABSTRACT_CHARS = 2000
SHARD_ROWS = 200_000
SCHEMA = pa.schema([("wikipedia_id", pa.string()), ("title", pa.string()), ("abstract", pa.string()),
                    ("n_paragraphs", pa.int32()), ("paragraphs", pa.list_(pa.string()))])


def page_row(d):
    paras = []
    for p in d["text"][1:]:  # text[0] is the title line
        p = p.strip()
        if p.startswith("BULLET::::"):
            p = p[len("BULLET::::"):].strip()
        if p and not p.startswith("Section::::"):
            paras.append(p)
    ab, n = [], 0
    for p in paras:
        if n >= ABSTRACT_CHARS:
            break
        ab.append(p); n += len(p) + 1
    return {"wikipedia_id": str(d["wikipedia_id"]), "title": d["wikipedia_title"], "abstract": " ".join(ab)[:ABSTRACT_CHARS],
            "n_paragraphs": len(paras), "paragraphs": paras}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kilt", default="/media/sda/data/kilt")
    ap.add_argument("--out", default="/media/sda/data/kilt/pages")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = os.path.join(a.kilt, "kilt_knowledgesource.json")
    buf, shard, total, t0 = [], 0, 0, time.time()

    def flush():
        nonlocal buf, shard
        if buf:
            pq.write_table(pa.Table.from_pylist(buf, schema=SCHEMA), os.path.join(a.out, "part-%04d.parquet" % shard), compression="zstd")
            shard += 1; buf = []

    with open(src, encoding="utf-8") as f:
        for line in f:
            buf.append(page_row(json.loads(line))); total += 1
            if len(buf) >= SHARD_ROWS:
                flush(); print("rows", total, "shards", shard, "%.0fs" % (time.time() - t0), flush=True)
    flush()
    sums = {}
    p = os.path.join(a.kilt, "SHA256SUMS")
    if os.path.exists(p):
        sums = {l.split()[1]: l.split()[0] for l in open(p) if l.strip()}
    json.dump({"rows": total, "shards": shard, "abstract_chars": ABSTRACT_CHARS, "source": src,
               "source_sha256": sums.get("kilt_knowledgesource.json"), "seconds": round(time.time() - t0, 1),
               "provenance": provenance()}, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print("DONE rows", total, "shards", shard)


if __name__ == "__main__":
    main()
