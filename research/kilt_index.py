"""Shared dense index over the KILT page table (research/PREREG_MILLION_SCALE.md): ONE index used by every arm that retrieves
(L1, L1', M, D), so no arm gets a retrieval advantage from its index.

Fixed before any tuning (no dev/test labels are read here):
  * index text per page: title + ". " + abstract, truncated by the model to 256 tokens (MiniLM's max_seq_length);
  * model: sentence-transformers/all-MiniLM-L6-v2, mean pooling, L2-normalised: the model MahaBodi's embedder
    (models/minilm) was exported from;
  * search: EXACT inner product (faiss.IndexFlatIP) over all pages, with no approximation.
Parity check: 2,000 seeded page texts are also embedded with MahaBodi's own ONNX embedder (Bodi.embed_text); the minimum
cosine between the two must be >= 0.999, or the script fails.

Writes <out>/emb.f16.npy (N x 384, float16), <out>/ids.npy (wikipedia_id order), <out>/MANIFEST.json (row count,
sha256 of emb/ids, parity stats, provenance).

    .venv/bin/python research/kilt_index.py --pages /media/sda/data/kilt/pages --out /media/sda/data/kilt/dense
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def page_text(title, abstract):
    return title + ". " + abstract


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="/media/sda/data/kilt/pages")
    ap.add_argument("--out", default="/media/sda/data/kilt/dense")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL, device=a.device)
    st.max_seq_length = 256
    shards = sorted(glob.glob(os.path.join(a.pages, "part-*.parquet")))
    n = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    emb = np.lib.format.open_memmap(os.path.join(a.out, "emb.f16.npy"), mode="w+", dtype=np.float16, shape=(n, 384))
    ids, i, t0 = [], 0, time.time()
    for s in shards:
        t = pq.read_table(s, columns=["wikipedia_id", "title", "abstract"]).to_pydict()
        texts = [page_text(ti, ab) for ti, ab in zip(t["title"], t["abstract"])]
        v = st.encode(texts, batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        emb[i:i + len(v)] = v.astype(np.float16); i += len(v); ids.extend(t["wikipedia_id"])
        print("embedded", i, "of", n, "%.0fs" % (time.time() - t0), flush=True)
    assert i == n
    emb.flush(); del emb
    np.save(os.path.join(a.out, "ids.npy"), np.array(ids))
    # parity vs MahaBodi's ONNX embedder on 2,000 seeded pages
    rng = np.random.default_rng(0); pick = np.sort(rng.choice(n, 2000, replace=False))
    E = np.load(os.path.join(a.out, "emb.f16.npy"), mmap_mode="r")
    allt = []
    for s in shards:
        t = pq.read_table(s, columns=["title", "abstract"]).to_pydict()
        allt.extend(zip(t["title"], t["abstract"]))
    texts = [page_text(*allt[j]) for j in pick]
    from mahabodi import Bodi
    b = Bodi(); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    M = np.array(b.embed_text(texts), dtype=np.float32)
    cos = np.sum(M * E[pick].astype(np.float32), 1) / np.linalg.norm(M, axis=1) / np.linalg.norm(E[pick].astype(np.float32), axis=1)
    parity = {"n": 2000, "min_cos": float(cos.min()), "mean_cos": float(cos.mean())}
    print("parity", parity, flush=True)
    json.dump({"rows": n, "dim": 384, "model": MODEL, "max_seq_length": 256, "index_text": "title + '. ' + abstract",
               "search": "exact inner product (faiss.IndexFlatIP), float32 at query time",
               "emb_sha256": sha(os.path.join(a.out, "emb.f16.npy")), "ids_sha256": sha(os.path.join(a.out, "ids.npy")),
               "pages_manifest": json.load(open(os.path.join(a.pages, "MANIFEST.json"))) if os.path.exists(os.path.join(a.pages, "MANIFEST.json")) else None,
               "parity_vs_mahabodi_embedder": parity, "seconds": round(time.time() - t0, 1), "provenance": provenance()},
              open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    assert parity["min_cos"] >= 0.999, "embedding parity failed: %s" % parity
    print("DONE")


if __name__ == "__main__":
    main()
