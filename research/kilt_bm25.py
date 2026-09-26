"""Shared BM25 index over the KILT page table (research/PREREG_MILLION_SCALE.md): the BM25 arm and the hard-negative
miner both use it.

Fixed before any tuning (no dev/test labels are read here): document text = title + ". " + abstract (the same text as
the dense index), bm25s defaults (Lucene BM25, k1 1.5, b 0.75), English stopwords, PyStemmer English stemmer.
Documents are indexed in the same order as the dense index (checked against dense/ids.npy when present).

    .venv/bin/python research/kilt_bm25.py --pages /media/sda/data/kilt/pages --out /media/sda/data/kilt/bm25
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="/media/sda/data/kilt/pages")
    ap.add_argument("--dense", default="/media/sda/data/kilt/dense")
    ap.add_argument("--out", default="/media/sda/data/kilt/bm25")
    a = ap.parse_args()
    import bm25s, Stemmer
    t0 = time.time()
    texts, ids = [], []
    for s in sorted(glob.glob(os.path.join(a.pages, "part-*.parquet"))):
        t = pq.read_table(s, columns=["wikipedia_id", "title", "abstract"]).to_pydict()
        texts.extend(ti + ". " + ab for ti, ab in zip(t["title"], t["abstract"])); ids.extend(t["wikipedia_id"])
    dp = os.path.join(a.dense, "ids.npy")
    same_order = bool(np.array_equal(np.load(dp), np.array(ids))) if os.path.exists(dp) else None
    if same_order is False:
        raise SystemExit("page order differs from the dense index")
    stem = Stemmer.Stemmer("english")
    tok = bm25s.tokenize(texts, stopwords="en", stemmer=stem, show_progress=False)
    del texts
    r = bm25s.BM25()
    r.index(tok, show_progress=False)
    r.save(a.out)
    h = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    json.dump({"rows": len(ids), "ids_sha256_joined": h, "same_order_as_dense": same_order, "text": "title + '. ' + abstract",
               "bm25": "bm25s defaults (k1 1.5, b 0.75, method lucene), stopwords en, PyStemmer english",
               "seconds": round(time.time() - t0, 1), "provenance": provenance()}, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print("DONE rows", len(ids), "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
