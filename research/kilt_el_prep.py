"""Entity-linking samples and hard-negative mining over the full KILT page table (research/PREREG_MILLION_SCALE.md).

Samples (seeded, fixed before any scoring):
- test: KILT AIDA dev (aidayago2-dev-kilt.jsonl), mentions whose gold wikipedia_id is in the page table, seeded shuffle
  (seed 0), first 1,000;
- dev: KILT AIDA train, same filter, seeded shuffle (seed 1), first 500. Used only to tune shortlist size k and option
  text length.
State per mention: the KILT input with its [START_ENT]/[END_ENT] markers, cut to +-200 characters around the mention.
Mining: for every state, the top-100 pages by exact dense inner product (the shared MiniLM index, dense/emb.f16.npy)
and by BM25 (the shared bm25s index). Written with the gold's row and its rank in each list. No arm is scored here.

    .venv/bin/python research/kilt_el_prep.py --out /media/sda/data/kilt/el
"""
import argparse, json, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

K = "/media/sda/data/kilt"
CTX = 200


def state_of(inp):
    a, b = inp.find("[START_ENT]"), inp.find("[END_ENT]")
    if a < 0 or b < 0:
        return inp[:2 * CTX]
    lo, hi = max(0, a - CTX), min(len(inp), b + len("[END_ENT]") + CTX)
    return inp[lo:hi].strip()


def sample(path, row_of, n, seed):
    rows, dropped = [], 0
    for line in open(path):
        d = json.loads(line)
        gid = d["output"][0]["provenance"][0]["wikipedia_id"] if d["output"] and d["output"][0].get("provenance") else None
        if gid is None or str(gid) not in row_of:
            dropped += 1; continue
        rows.append({"id": d["id"], "mention": d["meta"].get("mention"), "state": state_of(d["input"]), "gold_id": str(gid), "gold_row": row_of[str(gid)]})
    random.Random(seed).shuffle(rows)
    return rows[:n], {"eligible": len(rows), "dropped_gold_not_in_pages": dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(K, "el"))
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--query", choices=["state", "mention"], default="state", help="what the retrievers are queried with (clarification 2: mention)")
    ap.add_argument("--tag", default="", help="suffix for mentions<tag>.jsonl / MANIFEST<tag>.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = np.load(os.path.join(K, "dense", "ids.npy"))
    row_of = {str(x): i for i, x in enumerate(ids)}
    test, ts = sample(os.path.join(K, "aidayago2-dev-kilt.jsonl"), row_of, 1000, 0)
    dev, ds = sample(os.path.join(K, "aidayago2-train-kilt.jsonl"), row_of, 500, 1)
    allm = [("test", m) for m in test] + [("dev", m) for m in dev]
    states = [(m["state"] if a.query == "state" else (m["mention"] or m["state"])) for _, m in allm]
    t0 = time.time()
    import torch
    from sentence_transformers import SentenceTransformer
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev_); st.max_seq_length = 256
    Q = torch.tensor(st.encode(states, batch_size=256, normalize_embeddings=True, convert_to_numpy=True), device=dev_, dtype=torch.float16)
    E = np.load(os.path.join(K, "dense", "emb.f16.npy"), mmap_mode="r")
    best_s = torch.full((len(states), a.top), -1e4, device=dev_, dtype=torch.float32); best_i = torch.zeros((len(states), a.top), device=dev_, dtype=torch.long)
    CH = 500_000
    for s in range(0, E.shape[0], CH):
        blk = torch.tensor(np.asarray(E[s:s + CH]), device=dev_, dtype=torch.float16)
        sc = (Q @ blk.T).float()
        v, i = torch.topk(sc, a.top, dim=1)
        cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, i + s], 1)
        best_s, j = torch.topk(cs, a.top, dim=1); best_i = torch.gather(ci, 1, j)
        del blk, sc
    dense = best_i.cpu().numpy(); dense_s = best_s.cpu().numpy()
    t1 = time.time()
    import bm25s, Stemmer
    r = bm25s.BM25.load(os.path.join(K, "bm25"))
    tok = bm25s.tokenize(states, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)
    bres, bsc = r.retrieve(tok, k=a.top, show_progress=False, n_threads=8)
    t2 = time.time()
    for k, (split, m) in enumerate(allm):
        m["split"] = split
        m["dense_top"] = dense[k].tolist(); m["dense_scores"] = [round(float(x), 5) for x in dense_s[k]]
        m["bm25_top"] = [int(x) for x in bres[k]]; m["bm25_scores"] = [round(float(x), 4) for x in bsc[k]]
        m["gold_rank_dense"] = m["dense_top"].index(m["gold_row"]) + 1 if m["gold_row"] in m["dense_top"] else None
        m["gold_rank_bm25"] = m["bm25_top"].index(m["gold_row"]) + 1 if m["gold_row"] in m["bm25_top"] else None
    with open(os.path.join(a.out, "mentions%s.jsonl" % a.tag), "w") as f:
        for _, m in allm:
            f.write(json.dumps(m) + "\n")
    json.dump({"query": a.query, "test": ts, "dev": ds, "n_test": len(test), "n_dev": len(dev), "top": a.top, "context_chars": CTX,
               "dense_s": round(t1 - t0, 1), "bm25_s": round(t2 - t1, 1), "provenance": provenance()},
              open(os.path.join(a.out, "MANIFEST%s.json" % a.tag), "w"), indent=1)
    print("DONE", ts, ds)


if __name__ == "__main__":
    main()
