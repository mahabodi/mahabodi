"""The 1M-page constructed pool for the v2 scale run (research/PREREG_SCALE_V2.md, clarifications 5 and 5a).

One shared pool (store namespace `m1`), built by v1's rule (kilt_el_pools.py, PREREG_MILLION_SCALE pre-run clarification 1):
  1. golds: the fresh 1,000 (el_fresh_v2_ids.json, extracted as bench_el_store_v2.load_mentions("fresh")), the v1 test
     1,000 and the dev 500 (mentions_mq.jsonl);
  2. hard negatives: per mention, the dense and BM25 top-100 lists by mention query, interleaved by rank, gold excluded,
     the first h = min(100, floor(0.5 * (N - G) / M)) kept. Test and dev lists come from mentions_mq.jsonl (dense_top,
     bm25_top). The fresh mentions are mined here exactly as kilt_el_prep.py mines with --query mention: the
     sentence-transformers MiniLM embedding of the mention string against dense/emb.f16.npy (exact inner product in fp16,
     chunked top-100), and the shared bm25s index (top-100);
  3. nesting: the test 100K pool (pool_test_100000_mq.npy, the bridge's pool) is included;
  4. fill: random.Random(1_001_000) draws page rows from all pages not yet in the pool, up to exactly N = 1,000,000.

Outputs: <el>/pool_m1.npy (sorted page rows, int64), <el>/mentions_fresh_v2_mq.jsonl (the fresh mentions with their
mined lists, for audit and for --reuse-mined), and research/results/pool_m1.json.
SHA-256 convention: sha256 of the sorted rows as int64 little-endian bytes (np.asarray(rows, '<i8').tobytes()).

    KILT_DIR=~/data/kilt .venv/bin/python research/kilt_el_pool_1m.py
"""
import argparse, glob, hashlib, json, os, random, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

K = os.environ.get("KILT_DIR", "/media/sda/data/kilt")
EL = os.path.join(K, "el")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
N_POOL = 1_000_000
FILL_SEED = 1_001_000
TOP = 100
SETS = ("fresh", "test", "dev")


def norm_title(t):
    """v1's near-duplicate normalisation (kilt_el_pools.norm_title)."""
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", t.lower())).strip()


def compute_h(n, g, m):
    """h = min(100, floor(0.5 * (N - G) / M)), as kilt_el_pools ((N - G) // (2 M))."""
    return min(100, (n - g) // (2 * m))


def interleave(dense_top, bm25_top, gold):
    """v1's per-mention hard-negative order: dense and BM25 interleaved by rank, gold excluded, first occurrence kept."""
    inter, seen = [], set()
    for a, b in zip(dense_top, bm25_top):
        for x in (a, b):
            x = int(x)
            if x != gold and x not in seen:
                seen.add(x); inter.append(x)
    return inter


def rows_sha256(rows):
    return hashlib.sha256(np.asarray(sorted(int(r) for r in rows), dtype="<i8").tobytes()).hexdigest()


def build_pool(sets, titles, nest, n_pages, n=N_POOL, seed=FILL_SEED):
    """sets: {name: [mention dicts with id, gold_row, dense_top, bm25_top]}; titles: page titles by row (for the
    near-duplicate rate); nest: rows that must be in the pool. Returns (sorted rows, stats). Pure; no I/O."""
    golds = {s: {int(m["gold_row"]) for m in ms} for s, ms in sets.items()}
    G = set().union(*golds.values())
    M = sum(len(ms) for ms in sets.values())
    h = compute_h(n, len(G), M)
    hard, per_set, near_all = set(), {}, []
    for s, ms in sets.items():
        other = set().union(*(g for t, g in golds.items() if t != s))
        picks, near, other_picks, other_distinct, hs = 0, [], 0, set(), set()
        for m in ms:
            gold = int(m["gold_row"])
            picked = interleave(m["dense_top"], m["bm25_top"], gold)[:h]
            for x in picked:
                if x in other:
                    other_picks += 1; other_distinct.add(x)
                if norm_title(titles[x]) == norm_title(titles[gold]):
                    near.append((m["id"], titles[gold], titles[x]))
            picks += len(picked); hs.update(picked)
        hard |= hs
        rs = random.Random(5)
        per_set[s] = {"mentions": len(ms), "golds_distinct": len(golds[s]), "hard_negative_picks": picks,
                      "hard_negatives_distinct": len(hs), "near_duplicates": len(near),
                      "near_duplicate_rate": round(len(near) / max(1, h * len(ms)), 4),
                      "near_duplicate_examples": rs.sample(near, min(20, len(near))),
                      "other_set_golds_as_hard_negatives": {"picks": other_picks, "distinct": len(other_distinct)}}
        near_all += near
    nest = {int(x) for x in nest}
    pool = G | hard | nest
    n_before_fill = len(pool)
    if n_before_fill > n:
        raise ValueError("golds + hard negatives + nested pool = %d > N = %d" % (n_before_fill, n))
    rng = random.Random(seed)
    need = n - len(pool)
    while need > 0:
        x = rng.randrange(n_pages)
        if x not in pool:
            pool.add(x); need -= 1
    rows = np.array(sorted(pool), dtype=np.int64)
    assert len(rows) == n and G <= pool and nest <= pool
    hard_only = hard - G
    rs = random.Random(5)
    stats = {
        "N": n, "n_pages": n_pages, "mentions_M": M, "golds_G": len(G), "h": h,
        "golds_per_set": {s: len(g) for s, g in golds.items()},
        "gold_overlap": {"fresh_test": len(golds["fresh"] & golds["test"]) if "fresh" in golds and "test" in golds else None,
                         "fresh_dev": len(golds["fresh"] & golds["dev"]) if "fresh" in golds and "dev" in golds else None,
                         "test_dev": len(golds["test"] & golds["dev"]) if "test" in golds and "dev" in golds else None,
                         "all_three": len(set.intersection(*golds.values())) if len(golds) == 3 else None},
        "hard_negatives": len(hard_only), "hard_share": round(len(hard_only) / n, 4),
        "nested_rows": len(nest), "nested_rows_not_gold_or_hard": len(nest - G - hard),
        "random_fill": n - n_before_fill, "fill_seed": seed,
        "near_duplicates": len(near_all), "near_duplicate_rate": round(len(near_all) / max(1, h * M), 4),
        "near_duplicate_examples": rs.sample(near_all, min(20, len(near_all))),
        "per_set": per_set,
        "definitions": {
            "h": "min(100, (N - G) // (2 M)), G = distinct golds over the three sets, M = all mentions",
            "hard_share": "|hard negatives minus golds| / N",
            "near_duplicate": "a picked hard negative whose normalised title (lowercase, parentheses removed, whitespace "
                              "collapsed) equals its mention's gold's (v1, kilt_el_pools.py); rate = count / (h * mentions)",
            "other_set_golds_as_hard_negatives": "picks of this set's mentions that are another set's gold (picks: with "
                                                 "repetition; distinct: distinct rows)"}}
    return rows, stats


def mine(ms, top=TOP, chunk=500_000, qbatch=500):
    """kilt_el_prep.py's mining with --query mention: sentence-transformers all-MiniLM-L6-v2 (max_seq_length 256,
    normalised), fp16 exact inner product against dense/emb.f16.npy in 500K-row chunks with a running top-k; then the
    shared bm25s index (English stopwords, English Snowball stemmer, 8 threads). Queries run in groups of `qbatch`
    (kilt_el_prep ran all at once; per-row top-k does not depend on the other rows, and smaller groups bound memory)."""
    import torch
    from sentence_transformers import SentenceTransformer
    qtext = [m["mention"] or m["state"] for m in ms]
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU torch has no fast fp16 matmul (single-threaded, ~hours per query group), so on CPU the fp16 rows are upcast
    # exactly to fp32 and multiplied in fp32; mining_check below reports any reordering against the v1 GPU lists.
    cdt = torch.float16 if dev_ == "cuda" else torch.float32
    t0 = time.time()
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev_); st.max_seq_length = 256
    Qall = st.encode(qtext, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    E = np.load(os.path.join(K, "dense", "emb.f16.npy"), mmap_mode="r")
    dense, dense_s = [], []
    for q0 in range(0, len(qtext), qbatch):
        Q = torch.tensor(Qall[q0:q0 + qbatch], device=dev_, dtype=cdt)
        best_s = torch.full((len(Q), top), -1e4, device=dev_, dtype=torch.float32); best_i = torch.zeros((len(Q), top), device=dev_, dtype=torch.long)
        for s in range(0, E.shape[0], chunk):
            blk = torch.tensor(np.asarray(E[s:s + chunk]), device=dev_, dtype=cdt)
            sc = (Q @ blk.T).float()
            v, i = torch.topk(sc, top, dim=1)
            cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, i + s], 1)
            best_s, j = torch.topk(cs, top, dim=1); best_i = torch.gather(ci, 1, j)
            del blk, sc
        dense.append(best_i.cpu().numpy()); dense_s.append(best_s.cpu().numpy())
        print("mined dense", min(q0 + qbatch, len(qtext)), "/", len(qtext), round(time.time() - t0, 1), "s", flush=True)
    dense = np.concatenate(dense); dense_s = np.concatenate(dense_s)
    t1 = time.time()
    import bm25s, Stemmer
    r = bm25s.BM25.load(os.path.join(K, "bm25"))
    tok = bm25s.tokenize(qtext, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)
    bres, bsc = r.retrieve(tok, k=top, show_progress=False, n_threads=8)
    t2 = time.time()
    out = []
    for k, m in enumerate(ms):
        d = dict(m)
        d["dense_top"] = dense[k].tolist(); d["dense_scores"] = [round(float(x), 5) for x in dense_s[k]]
        d["bm25_top"] = [int(x) for x in bres[k]]; d["bm25_scores"] = [round(float(x), 4) for x in bsc[k]]
        d["gold_rank_dense"] = d["dense_top"].index(d["gold_row"]) + 1 if d["gold_row"] in d["dense_top"] else None
        d["gold_rank_bm25"] = d["bm25_top"].index(d["gold_row"]) + 1 if d["gold_row"] in d["bm25_top"] else None
        out.append(d)
    info = {"query": "mention (m['mention'] or m['state'])", "top": top, "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "max_seq_length": 256, "device": dev_, "dense_dtype": "%s matmul over float16-stored rows, top-k in float32" % str(cdt).replace("torch.", ""),
            "torch_threads": torch.get_num_threads(), "dense_chunk_rows": chunk,
            "query_group": qbatch, "bm25": "bm25s.BM25.load(<kilt>/bm25), stopwords=en, Stemmer english, n_threads=8",
            "dense_s": round(t1 - t0, 1), "bm25_s": round(t2 - t1, 1)}
    return out, info


def mining_check(mined, ref):
    """Re-mined v1 mentions vs their mentions_mq.jsonl lists (descriptive: a different machine or device can reorder ties)."""
    by = {m["id"]: m for m in ref}
    res = {}
    for key in ("dense_top", "bm25_top"):
        same = [m[key] == by[m["id"]][key] for m in mined]
        ov = [len(set(m[key]) & set(by[m["id"]][key])) / max(1, len(by[m["id"]][key])) for m in mined]
        res[key] = {"identical_lists": int(sum(same)), "mean_set_overlap": round(float(np.mean(ov)), 4), "min_set_overlap": round(float(np.min(ov)), 4)}
    res["items"] = len(mined)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-mined", action="store_true", help="read the fresh lists from <el>/mentions_fresh_v2_mq.jsonl instead of mining")
    ap.add_argument("--check-v1", type=int, default=100, help="also re-mine the first N test and first N dev mentions and compare with mentions_mq.jsonl (0: skip)")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from bench_el_store_v2 import load_mentions
    t0 = time.time()
    M = [json.loads(l) for l in open(os.path.join(EL, "mentions_mq.jsonl"))]
    sets = {"test": [m for m in M if m["split"] == "test"], "dev": [m for m in M if m["split"] == "dev"]}
    assert len(sets["test"]) == 1000 and len(sets["dev"]) == 500
    fresh_path = os.path.join(EL, "mentions_fresh_v2_mq.jsonl")
    if a.reuse_mined:
        fresh = [json.loads(l) for l in open(fresh_path)]
        mined_info = json.load(open(fresh_path + ".info.json"))
        mined_info["reused_from"] = fresh_path
    else:
        fresh_ms = load_mentions("fresh")
        check = sets["test"][:a.check_v1] + sets["dev"][:a.check_v1]
        mined, mined_info = mine(fresh_ms + [{k: m[k] for k in ("id", "mention", "state", "gold_row")} for m in check])
        fresh = mined[:len(fresh_ms)]
        if check:
            mined_info["v1_remining_check"] = mining_check(mined[len(fresh_ms):], check)
            print("v1 re-mining check", mined_info["v1_remining_check"], flush=True)
        with open(fresh_path + ".tmp", "w") as f:
            for m in fresh:
                f.write(json.dumps(m) + "\n")
        os.replace(fresh_path + ".tmp", fresh_path)
        json.dump(mined_info, open(fresh_path + ".info.json", "w"), indent=1)
    ids = json.load(open(os.path.join(R, "el_fresh_v2_ids.json")))
    assert sorted(m["id"] for m in fresh) == sorted(ids) and len(fresh) == 1000
    sets = {"fresh": fresh, **sets}
    titles = []
    for s in sorted(glob.glob(os.path.join(K, "pages", "part-*.parquet"))):
        titles.extend(pq.read_table(s, columns=["title"]).column("title").to_pylist())
    nest = np.load(os.path.join(EL, "pool_test_100000_mq.npy")).tolist()
    rows, stats = build_pool(sets, titles, nest, len(titles))
    np.save(os.path.join(EL, "pool_m1.npy"), rows)
    out = {"prereg": "research/PREREG_SCALE_V2.md clarifications 5 and 5a", "pool_file": os.path.join(EL, "pool_m1.npy"),
           "rows_sha256": rows_sha256(rows), "sha256_convention": "sha256 of the sorted page rows as int64 little-endian bytes",
           "fresh_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
           "nested_pool": "pool_test_100000_mq.npy", "textless_page_610897_in_pool": bool(610897 in set(rows.tolist())),
           "label": "1M-page constructed pool (golds + mined hard negatives + random fill)",
           **stats, "fresh_mining": {**mined_info, "file": fresh_path}, "seconds": round(time.time() - t0, 1), "provenance": provenance()}
    json.dump(out, open(os.path.join(R, "pool_m1.json"), "w"), indent=1, default=str)
    print("POOL-DONE", {k: v for k, v in out.items() if k in ("rows_sha256", "N", "golds_G", "h", "hard_negatives", "hard_share", "random_fill", "near_duplicate_rate", "gold_overlap")}, flush=True)


if __name__ == "__main__":
    main()
