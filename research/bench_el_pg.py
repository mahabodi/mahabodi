"""EL test, MahaBodi arms through the PostgreSQL store (PREREG_MILLION_SCALE.md clarifications 4, 4b, 4c).

The shortlist comes from deploy/sync/mahabodi_pg.py::search (lexical + typo-corrected lexical + dense pgvector, fused by
reciprocal rank), queried by the mention string. 3k passages are mapped to pages in rank order and truncated to k, as in
in-process M. The decision step is unchanged, so only candidate generation differs from in-process M.
  M_pg       PG shortlist -> MahaBodi `decide`          ("MahaBodi via PG store", not the in-process M)
  L1p_pg     the same shortlist -> Laya `predict`
  MA_pg      alias-memory hits, then the PG shortlist, seeded-shuffled (5a) -> `decide`   (alias memory as in bench_el.py)
  L1p_MA_pg  MA_pg's shuffled shortlist -> Laya `predict`; *_rank_order: the same in rank order (ablation)
Databases: el_t100k (exactly the test 100K pool: the bridge vs in-process M) and el_full (5.9M pages). (k, n) = M's
dev-selected setting from bench_el_tune.json. Per-mention checkpoint; per-item predictions in the output.

    .venv/bin/python research/bench_el_pg.py --db el_t100k,el_full   -> research/results/bench_el_pg.json
"""
import argparse, json, os, random, re, sys, time, zlib
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "sync"))
from provenance import provenance  # noqa: E402
from bench import wilson  # noqa: E402
from bench_el import Pages, laya_choice, K, EL, R, ROOT  # noqa: E402
import mahabodi_pg  # noqa: E402

DSN = "host=127.0.0.1 port=5433 user=postgres password=mahabodi dbname=%s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="el_t100k,el_full")
    ap.add_argument("--laya", default=os.path.join(ROOT, "models", "laya-v2"))
    ap.add_argument("--tag", default="_mq")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N mentions only (results not for reporting)")
    ap.add_argument("--split", default="test", choices=["test", "dev"], help="dev: mechanics smoke test only (dev gold is not in the test pool)")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    from mahabodi import Bodi
    from kilt_el_prep import state_of
    Mall = [json.loads(l) for l in open(os.path.join(EL, "mentions%s.jsonl" % a.tag))]
    ms = [m for m in Mall if m["split"] == a.split]
    if a.limit:
        ms = ms[:a.limit]
    dev_ids = {m["id"] for m in Mall if m["split"] == "dev"}
    tune = json.load(open(os.path.join(R, "bench_el_tune.json")))
    km, nm = tune["M"]["k"], tune["M"]["n"]
    P = Pages()
    ids = np.load(os.path.join(K, "dense", "ids.npy")); row_of = {str(x): i for i, x in enumerate(ids)}
    # alias memory: identical construction to bench_el.py (AIDA train, dev mentions excluded, no counts in the text)
    train = []
    for line in open(os.path.join(K, "aidayago2-train-kilt.jsonl")):
        d = json.loads(line)
        if d["id"] in dev_ids or not d["output"] or not d["output"][0].get("provenance"):
            continue
        train.append((d["meta"].get("mention") or "", state_of(d["input"]), str(d["output"][0]["provenance"][0]["wikipedia_id"])))
    pairs = sorted({(men.strip(), row_of[g]) for men, _, g in train if men.strip() and g in row_of})
    alias_rows = [r for _, r in pairs]
    am = Bodi(); am.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    am.ingest_batch([{"text": "# %s\n\nrefers to: %s" % (men, P.titles[r]), "source": "al%d" % i} for i, (men, r) in enumerate(pairs)])
    lb = Bodi(); lb.load_laya(a.laya, intra_threads=8)
    b = Bodi(); b.load_laya(a.laya, intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    gold = [m["gold_row"] for m in ms]
    out = {"question_arms": "M_pg, L1p_pg, MA_pg, L1p_MA_pg (clarifications 4, 4b, 4c)", "n": len(ms), "settings": {"k": km, "n": nm},
           "mention_ids": [m["id"] for m in ms], "gold_rows": gold, "provenance": provenance(), "alias_records": len(pairs),
           "store": "PostgreSQL 17 + pgvector (deploy/postgres/Dockerfile); snapshot passages; page vector on first passage; "
                    "HNSW m=16 ef_construction=64, hnsw.ef_search=100; AGE graph not loaded (search does not read it)",
           "dbs": {}}
    for db in a.db.split(","):
        pool = None if db == "el_full" else set(np.load(os.path.join(EL, "pool_test_%s%s.npy" % ({"el_t100k": "100000", "el_t10k": "10000"}[db], a.tag))).tolist())
        inpool = (lambda r: True) if pool is None else (lambda r: r in pool)
        c = mahabodi_pg.connect(DSN % db); c.execute("SET hnsw.ef_search = 100")
        ckp = os.path.join(R, "bench_el_pg_%s%s.ckpt.json" % (db, a.out_suffix))
        ck = json.load(open(ckp)) if os.path.exists(ckp) else {}
        for i, m in enumerate(ms):
            if m["id"] in ck:
                continue
            q = m["mention"] or m["state"]
            t = time.perf_counter()
            qv = b.embed_text([q])[0]
            hits = mahabodi_pg.search(c, "kilt", q, qvec=qv, k=3 * km, model="minilm")
            sh = []
            for h in hits:
                mm = re.match(r"pg_(\d+)_", h["id"])
                if mm and int(mm.group(1)) not in sh:
                    sh.append(int(mm.group(1)))
            sh = sh[:km]
            t_ret = (time.perf_counter() - t) * 1000
            p_m = laya_choice(b, m["state"], sh, P, nm, decide=True)
            t_m = (time.perf_counter() - t) * 1000
            ma = []
            for h in am.query(q, k=3 * km).get("hits", []):
                mm = re.match(r"F_al_(\d+)_", h["id"])
                if mm:
                    r = alias_rows[int(mm.group(1))]
                    if inpool(r) and r not in ma:
                        ma.append(r)
            for r in sh:
                if r not in ma:
                    ma.append(r)
            ma = ma[:km]
            # clarification 5a: MA arms decide over a seeded shuffle of their shortlist (primary); rank order is the ablation
            mas = list(ma); random.Random(zlib.crc32(m["id"].encode())).shuffle(mas)
            ck[m["id"]] = {"shortlist": sh, "M_pg": p_m, "L1p_pg": laya_choice(lb, m["state"], sh, P, nm, decide=False),
                           "MA_shortlist": ma, "MA_pg": laya_choice(b, m["state"], mas, P, nm, decide=True),
                           "L1p_MA_pg": laya_choice(lb, m["state"], mas, P, nm, decide=False),
                           "MA_pg_rank_order": laya_choice(b, m["state"], ma, P, nm, decide=True),
                           "L1p_MA_pg_rank_order": laya_choice(lb, m["state"], ma, P, nm, decide=False),
                           "retrieval_ms": round(t_ret, 1), "M_pg_ms": round(t_m, 1)}
            if i % 20 == 0 or i == len(ms) - 1:
                json.dump(ck, open(ckp + ".tmp", "w")); os.replace(ckp + ".tmp", ckp)
                print(db, i + 1, "/", len(ms), flush=True)
        json.dump(ck, open(ckp + ".tmp", "w")); os.replace(ckp + ".tmp", ckp)
        D = {"N_pages": 5903530 if pool is None else len(pool), "arms": {}}
        rows = [ck[m["id"]] for m in ms]
        for arm in ("M_pg", "L1p_pg", "MA_pg", "L1p_MA_pg", "MA_pg_rank_order", "L1p_MA_pg_rank_order"):
            cor = [r[arm] == g for r, g in zip(rows, gold)]
            D["arms"][arm] = {"accuracy": round(float(np.mean(cor)), 4), "ci95": wilson(sum(cor), len(cor)),
                              "answered": sum(r[arm] is not None for r in rows), "pred": [r[arm] for r in rows]}
        for key, arm in (("shortlist", "M_pg"), ("MA_shortlist", "MA_pg")):
            hit = [g in r[key] for r, g in zip(rows, gold)]
            D[arm + "_shortlist_recall"] = round(float(np.mean(hit)), 4)
            D[arm + "_accuracy_given_gold_in_shortlist"] = round(float(np.mean([r[arm] == g for r, g, h in zip(rows, gold, hit) if h])), 4) if any(hit) else None
        D["retrieval_ms_p50"] = float(np.median([r["retrieval_ms"] for r in rows]))
        D["M_pg_latency_ms_p50"] = float(np.median([r["M_pg_ms"] for r in rows]))
        D["n_atfs"] = c.execute("SELECT count(*) FROM mahabodi.atf WHERE namespace = 'kilt'").fetchone()[0]
        out["dbs"][db] = D
        print("DB-DONE", db, flush=True)
    json.dump(out, open(os.path.join(R, "bench_el_pg%s.json" % a.out_suffix), "w"), indent=1, default=str)
    print("PG-DONE", flush=True)


if __name__ == "__main__":
    main()
