"""Load KILT pages into the MahaBodi PostgreSQL store for the full-stage EL arms (PREREG_MILLION_SCALE.md clarification 4b).

Two databases on one server, each with deploy/postgres/01_schema.sql and namespace `kilt`:
  el_full   all 5,903,530 pages
  el_t100k  exactly the test 100K pool (the bridge vs in-process M)
  el_t10k   exactly the test 10K pool (the bridge fallback, clarification 5a)
Rows are the Bodi.snapshot() passages of `# <title>\n\n<abstract>` (clarification 4c: MahaBodi splits a page into
passages; a one-row-per-page loader failed the parity check). The page's existing KILT MiniLM vector is attached to its
first passage. Every step is resumable: finished shards are recorded in <state>/done.json.

    .venv/bin/python research/el_pg_load.py load|vocab|index --db el_full|el_t100k
"""
import argparse, glob, io, json, os, re, sys, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "sync"))
import psycopg  # noqa: E402
from mahabodi_pg import _terms  # noqa: E402  (the reference store's vocabulary rule)

K = os.environ.get("KILT_DIR", "/media/sda/data/kilt"); EL = os.path.join(K, "el")
STATE = os.environ.get("PG_EL_STATE", "/media/sda/pg_el/state")
DSN = "host=127.0.0.1 port=5433 user=postgres password=mahabodi dbname=%s"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shards():
    out, n = [], 0
    for s in sorted(glob.glob(os.path.join(K, "pages", "part-*.parquet"))):
        m = pq.ParquetFile(s).metadata.num_rows
        out.append((s, n, n + m)); n += m
    return out


def body(title, abstract):
    return "%s: %s" % (title, abstract)


def conn(db):
    c = psycopg.connect(DSN % db, autocommit=True)
    c.execute("LOAD 'age'"); c.execute("SET search_path = ag_catalog, mahabodi, public")
    return c


def ensure_db(db):
    c = psycopg.connect(DSN % "postgres", autocommit=True)
    if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,)).fetchone():
        c.execute('CREATE DATABASE "%s"' % db)
        d = psycopg.connect(DSN % db, autocommit=True)
        d.execute(open(os.path.join(ROOT, "deploy", "postgres", "01_schema.sql")).read())
        d.execute("LOAD 'age'"); d.execute("SET search_path = ag_catalog, mahabodi, public")
        d.execute("SELECT mahabodi.create_namespace('kilt', 'sentence-transformers/all-MiniLM-L6-v2', 384)")
        # bulk load: build the search indexes after the rows are in (step `index`)
        d.execute("DROP INDEX IF EXISTS mahabodi.atf_tsv_gin"); d.execute("DROP INDEX IF EXISTS mahabodi.atf_embedding_hnsw")


def rows_for(db):
    return None if db == "el_full" else set(np.load(os.path.join(EL, "pool_test_%s_mq.npy" % {"el_t100k": "100000", "el_t10k": "10000"}[db])).tolist())


def _snapshot(docs):
    """Worker process: MahaBodi's own passage split of one batch (clarification 4c)."""
    from mahabodi import Bodi
    bo = Bodi()
    bo.ingest_batch(docs)
    return bo.snapshot()


def load(db, sub=50000):
    """Clarification 4c: the rows are Bodi.snapshot() ATFs of each 50K-page batch, copied unchanged; the page vector goes
    on the page's first passage (the one carrying the `<title>: ` prefix, else the first in snapshot order).
    Pipelined: a worker process snapshots batch i+1 while this process copies batch i into the store."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    os.makedirs(STATE, exist_ok=True)
    ensure_db(db)
    keep = rows_for(db)
    dp = os.path.join(STATE, "done_%s.json" % db)
    done = json.load(open(dp)) if os.path.exists(dp) else {}
    E = np.load(os.path.join(K, "dense", "emb.f16.npy"), mmap_mode="r")
    c = conn(db)

    def batches():
        for s, a, b_ in shards():
            rs_all = [r for r in range(a, b_) if keep is None or r in keep]
            todo = [j for j in range(0, len(rs_all), sub) if "%s:%d" % (os.path.basename(s), j) not in done]
            if not todo:
                continue
            t = pq.read_table(s, columns=["title", "abstract"])
            ti, ab = t.column("title").to_pylist(), t.column("abstract").to_pylist()
            for j in todo:
                rs = rs_all[j:j + sub]
                yield ("%s:%d" % (os.path.basename(s), j), rs, {r: ti[r - a] for r in rs},
                       [{"text": "# %s\n\n%s" % (ti[r - a], ab[r - a]), "source": "pg%d" % r} for r in rs])

    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        it = batches()
        nxt = next(it, None)
        fut = ex.submit(_snapshot, nxt[3]) if nxt else None
        while nxt:
            t0 = time.time()
            key, rs, title, _ = nxt
            snap = fut.result()
            nxt = next(it, None)
            fut = ex.submit(_snapshot, nxt[3]) if nxt else None  # the next batch splits while this one is copied
            texts = snap["texts"]; first = {}
            for x in snap["atfs"]:
                r = int(re.match(r"pg_(\d+)_", x["id"]).group(1))
                if r not in first or (not texts[first[r]].startswith(title[r] + ": ") and texts[x["id"]].startswith(title[r] + ": ")):
                    first[r] = x["id"]
            # a batch commits atomically before it is recorded; a crash between the two leaves it in the store: skip it
            if c.execute("SELECT 1 FROM mahabodi.atf WHERE namespace = 'kilt' AND id = %s", (snap["atfs"][0]["id"],)).fetchone():
                print("already in store (committed before a crash):", key, flush=True)
            else:
                with c.transaction():
                    with c.cursor().copy("COPY mahabodi.atf (namespace, id, action, input, logic, access, events, data_connections, body) FROM STDIN") as cp:
                        for x in snap["atfs"]:
                            cp.write_row(("kilt", x["id"], x.get("action", ""), x.get("input", ""), x.get("logic", ""), x.get("access", ""),
                                          x.get("events", ""), list(x.get("data_connections", [])), texts.get(x["id"], "")))
                    with c.cursor().copy("COPY mahabodi.atf_embedding (namespace, atf_id, model, embedding) FROM STDIN") as cp:
                        for r, i in first.items():
                            cp.write_row(("kilt", i, "minilm", "[" + ",".join("%.7g" % v for v in np.asarray(E[r], dtype=np.float32)) + "]"))
            done[key] = {"pages": len(rs), "atfs": len(snap["atfs"]), "vectors": len(first)}
            json.dump(done, open(dp + ".tmp", "w")); os.replace(dp + ".tmp", dp)
            print("loaded", db, key, done[key], round(time.time() - t0, 1), "s", flush=True)
    tot = {k: sum(v[k] for v in done.values()) for k in ("pages", "atfs", "vectors")}
    n_pg = c.execute("SELECT count(*) FROM mahabodi.atf WHERE namespace = 'kilt'").fetchone()[0]
    print("LOAD-DONE", db, tot, "rows in store", n_pg, flush=True)
    if n_pg != tot["atfs"]:
        sys.exit("LOAD-MISMATCH: store %d vs snapshots %d" % (n_pg, tot["atfs"]))


def vocab(db):
    # the reference loader's vocabulary: terms of id + action + body, document frequency
    c = conn(db)
    if c.execute("SELECT count(*) FROM mahabodi.vocab WHERE namespace = 'kilt'").fetchone()[0]:
        print("VOCAB-DONE (already present)", flush=True); return
    df = {}
    with c.cursor(name="v") as cur:
        cur.itersize = 20000
        cur.execute("SELECT id, action, body FROM mahabodi.atf WHERE namespace = 'kilt'")
        for i, (id_, act, bd) in enumerate(cur):
            for w in _terms(" ".join([id_, act, bd])):
                df[w] = df.get(w, 0) + 1
            if i % 500000 == 0:
                print("vocab", i, len(df), flush=True)
    with c.transaction():
        with c.cursor().copy("COPY mahabodi.vocab (namespace, term, doc_freq) FROM STDIN") as cp:
            for w, n in df.items():
                cp.write_row(("kilt", w, n))
    print("VOCAB-DONE", len(df), flush=True)


def index(db):
    c = conn(db)
    c.execute("SET maintenance_work_mem = '%s'" % os.environ.get("PG_MWM", "4GB")); c.execute("SET max_parallel_maintenance_workers = %d" % int(os.environ.get("PG_WORKERS", "6")))
    for name, sql in (("atf_tsv_gin", "CREATE INDEX IF NOT EXISTS atf_tsv_gin ON mahabodi.atf USING gin (tsv)"),
                      ("atf_embedding_hnsw", "CREATE INDEX IF NOT EXISTS atf_embedding_hnsw ON mahabodi.atf_embedding "
                                             "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")):
        t0 = time.time(); c.execute(sql); print("index", db, name, round(time.time() - t0, 1), "s", flush=True)
    c.execute("VACUUM ANALYZE mahabodi.atf"); c.execute("VACUUM ANALYZE mahabodi.atf_embedding")
    print("INDEX-DONE", db, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["load", "vocab", "index"])
    ap.add_argument("--db", default="el_t100k", choices=["el_full", "el_t100k", "el_t10k"])
    x = ap.parse_args()
    os.makedirs(STATE, exist_ok=True)
    {"load": lambda: load(x.db), "vocab": lambda: vocab(x.db), "index": lambda: index(x.db)}[x.step]()
