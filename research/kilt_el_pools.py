"""Entity-linking candidate pools per stage (research/PREREG_MILLION_SCALE.md, pre-run clarification 1).

Per split (test, dev) and stage N in {10000, 100000}: golds + per-mention hard negatives (dense and BM25 top lists
interleaved by rank, gold excluded, h = min(100, floor(0.5*(N-G)/M))) + seeded random fill to N. Nested: the 100K pool
contains the 10K pool. The full stage is all 5,903,530 pages (no file). The other split's golds are excluded from the
fill; any unavoidable overlap (another split's gold appearing as a hard negative) is counted.

Reports per stage: h, hard-negative share, near-duplicate rate (a hard negative whose normalised title equals the gold's)
with 20 random examples. Writes <el>/pool_<split>_<N>.npy (sorted page rows) and <el>/POOLS.json.

    .venv/bin/python research/kilt_el_pools.py
"""
import json, os, random, re, sys
import numpy as np, pyarrow.parquet as pq, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

K = "/media/sda/data/kilt"
EL = os.path.join(K, "el")
STAGES = [10_000, 100_000]


def norm_title(t):
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", t.lower())).strip()


def main():
    M = [json.loads(l) for l in open(os.path.join(EL, "mentions.jsonl"))]
    titles = []
    for s in sorted(glob.glob(os.path.join(K, "pages", "part-*.parquet"))):
        titles.extend(pq.read_table(s, columns=["title"]).column("title").to_pylist())
    NP = len(titles)
    out = {"n_pages": NP, "stages": {}, "provenance": provenance()}
    golds = {sp: sorted({m["gold_row"] for m in M if m["split"] == sp}) for sp in ("test", "dev")}
    for sp in ("test", "dev"):
        ms = [m for m in M if m["split"] == sp]
        other = set(golds["dev" if sp == "test" else "test"])
        G = set(golds[sp])
        prev = set()
        for N in STAGES:
            h = min(100, (N - len(G)) // (2 * len(ms)))
            hard, near, overlap = set(), [], 0
            for m in ms:
                inter, seen = [], set()
                for a, b in zip(m["dense_top"], m["bm25_top"]):
                    for x in (a, b):
                        if x != m["gold_row"] and x not in seen:
                            seen.add(x); inter.append(x)
                picked = inter[:h]
                for x in picked:
                    if x in other:
                        overlap += 1
                    if norm_title(titles[x]) == norm_title(titles[m["gold_row"]]):
                        near.append((m["id"], titles[m["gold_row"]], titles[x]))
                hard.update(picked)
            pool = set(G) | hard | prev
            rng = random.Random(1000 + N + (0 if sp == "test" else 7))
            need = N - len(pool)
            cand_excl = pool | other
            while need > 0:
                x = rng.randrange(NP)
                if x not in cand_excl:
                    pool.add(x); cand_excl.add(x); need -= 1
            arr = np.array(sorted(pool), dtype=np.int64)
            assert len(arr) == N and prev <= pool
            np.save(os.path.join(EL, "pool_%s_%d.npy" % (sp, N)), arr)
            nh = len(hard - G)
            rs = random.Random(5)
            out["stages"]["%s_%d" % (sp, N)] = {"N": N, "golds": len(G), "h": h, "hard_negatives": nh, "hard_share": round(nh / N, 4),
                                               "near_duplicates": len(near), "near_duplicate_rate": round(len(near) / max(1, h * len(ms)), 4),
                                               "near_duplicate_examples": rs.sample(near, min(20, len(near))),
                                               "other_split_golds_as_hard_negatives": overlap}
            print(sp, N, {k: v for k, v in out["stages"]["%s_%d" % (sp, N)].items() if k != "near_duplicate_examples"}, flush=True)
            prev = pool
    json.dump(out, open(os.path.join(EL, "POOLS.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
