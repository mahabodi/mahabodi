"""PREREG_SCALE_V2 step 3 (and the full-scale part of step 2): entity linking with the shipped PostgreSQL store.

Phases (each writes its own result file; each resumable):
  bridge      test 100K pool, v1 test mentions: store M (exact dense, then IVFFlat at the dev-chosen probes) vs in-process M
  load_full   all 5,903,530 pages into namespace `full` (resumable by batch), build, ONE density pass (24 h budget),
              IVFFlat index (lists = ceil(sqrt(rows)), maintenance_work_mem 10GB)  [stopped at 4.14M pages: clarification 5]
  partial_report  the stopped load_full as a measured result (5a.6): batches/pages done, elapsed time from the load log,
              per-table sizes and per-namespace rows for `full`, bytes per page, stop reason. Must run before drop_full.
  drop_full   drops the whole mahabodi_store schema (clarification 5b; needs the partial report)
  load_1m     clarification 5: the 1M constructed pool (el/pool_m1.npy) into namespace `m1`, build, ONE density pass,
              IVFFlat (lists = ceil(sqrt(rows)), maintenance_work_mem 10GB, 6 workers); sizes and bytes per page
  build_el_m1 clarification 5a.3: a v1-layout database el_m1 holding the pool pages' v1 rows copied from el_full
              (page atf rows + first-passage vectors), vocabulary recomputed, v1's GIN and HNSW indexes
  probes      full-scale probes on the 500 DEV mentions vs an exact NumPy scan of the namespace's stored passage vectors
              (--namespace, default m1)
  primary     fresh 1,000 (el_fresh_v2_ids.json) and the v1 test 1,000 on `m1`: store M vs L1 (Laya + dense top-20 among
              the pool pages, n = 16) and P0 (bench_el's alias prior, then normalised title match, within the pool)
  v1check     the v1 corner (mahabodi_pg.search on el_full, unchanged) on the v1 test items must equal bench_el_pg.json
  ablation    fresh items on `m1`: {per-passage, first-passage (v1) vectors} x {MahaBodi cascade, PG text ranking},
              RRF k = 60; the v1 lists come from el_m1 (5a.3)

Shortlist rule everywhere: k = 60 hits -> pages (pg_<row>_...) in rank order -> first 20; decider `decide`, n = 48.

    KILT_DIR=... python research/bench_el_store_v2.py --phase bridge --dsn "..."
"""
import argparse, glob, hashlib, json, math, os, re, sys, time, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from store_parity_gate import wilson, mcnemar  # noqa: E402  (verbatim copies of bench.py's; registers the light 'bench')
from bench_el import Pages, laya_choice, ntitle, EL, R, ROOT, K  # noqa: E402

V1_DSN_DB = "el_full"
PAGE = re.compile(r"(?:F_)?pg_(\d+)_")
N_ALL_PAGES = 5903530
LOAD_BATCH = 20000  # load_pool's batch
FULL_PARTIAL = os.path.join(R, "el_store_v2_full_partial.json")


def no_decision_cache(b):
    """Clarification 3e: every decide call runs with the decision cache off (predict has no cache)."""
    from mahabodi import Bodi
    b.decide = lambda state, questions, **o: Bodi.decide(b, state, questions, **{**o, "cache": False})
    return b


def pages_of(ids, k=20):
    rows = []
    for i in ids:
        m = PAGE.match(i)
        if m and int(m.group(1)) not in rows:
            rows.append(int(m.group(1)))
    return rows[:k]


def rrf(lex, dense, k=50):
    """query_with's / mahabodi_pg.search's fusion: lexical top 50 + dense top 50, 1/(60 + rank), ties by id. Scores are exact
    fractions (SQL sums NUMERIC exactly), so rational ties resolve by id as in SQL; ids sort by code point, which equals
    the databases' C.UTF-8 collation for these ASCII ids (checked: el_full and the store DB are C.UTF-8)."""
    from fractions import Fraction
    s = {}
    for r, i in enumerate(lex[:k]):
        s[i] = s.get(i, Fraction(0)) + Fraction(1, 60 + r + 1)
    for r, i in enumerate(dense[:k]):
        s[i] = s.get(i, Fraction(0)) + Fraction(1, 60 + r + 1)
    return [i for i, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))]


def extract(path, ids):
    """Mentions for `ids` from a KILT EL file, exactly as kilt_el_prep.sample builds them (mention query, +-200-char state)."""
    from kilt_el_prep import state_of
    dense_ids = np.load(os.path.join(K, "dense", "ids.npy")); row_of = {str(x): i for i, x in enumerate(dense_ids)}
    out = {}
    for line in open(path):
        d = json.loads(line)
        if d["id"] in ids:
            g = str(d["output"][0]["provenance"][0]["wikipedia_id"])
            out[d["id"]] = {"id": d["id"], "mention": d["meta"].get("mention"), "state": state_of(d["input"]), "gold_row": row_of[g]}
    return out


def extractor_self_check():
    """Reviewer's check: the v2 extractor reproduces mentions_mq.jsonl (v1 test from aidayago2-dev, dev from aidayago2-train)."""
    M = [json.loads(l) for l in open(os.path.join(EL, "mentions_mq.jsonl"))]
    res = {}
    for split, f in (("test", "aidayago2-dev-kilt.jsonl"), ("dev", "aidayago2-train-kilt.jsonl")):
        ms = [m for m in M if m["split"] == split]
        got = extract(os.path.join(K, f), {m["id"] for m in ms})
        same = sum(1 for m in ms if m["id"] in got and all(got[m["id"]][k] == m[k] for k in ("mention", "state", "gold_row")))
        res[split] = {"items": len(ms), "identical": same}
    res["ok"] = all(v["items"] == v["identical"] for v in res.values() if isinstance(v, dict))
    return res


def load_mentions(which):
    M = [json.loads(l) for l in open(os.path.join(EL, "mentions_mq.jsonl"))]
    if which in ("dev", "test"):
        return [m for m in M if m["split"] == which]
    ids = set(json.load(open(os.path.join(R, "el_fresh_v2_ids.json"))))
    out = sorted(extract(os.path.join(K, "aidayago2-dev-kilt.jsonl"), ids).values(), key=lambda m: m["id"])
    assert len(out) == 1000
    return out


def score(preds, hits, gold, extra=None):
    c = [p == g for p, g in zip(preds, gold)]
    d = {"accuracy": round(float(np.mean(c)), 4), "ci95": wilson(sum(c), len(c)), "shortlist_recall": round(float(np.mean(hits)), 4),
         "pred": preds, "shortlist_hit": hits}
    d.update(extra or {})
    return d


def latency_block(ms_list, resumed, prefix="latency_ms"):
    """Clarification 3d: latency over the scored pass in mention order, no separate warm-up. Reported: the first query
    (cold), p50/p95 over all queries, and p50/p95 over the second half (queries n/2 .. n-1 in order)."""
    a = np.asarray(ms_list, dtype=float); h = a[len(a) // 2:]
    return {prefix + "_first": round(float(a[0]), 1), prefix + "_p50": round(float(np.median(a)), 1), prefix + "_p95": round(float(np.percentile(a, 95)), 1),
            prefix + "_second_half_p50": round(float(np.median(h)), 1), prefix + "_second_half_p95": round(float(np.percentile(h, 95)), 1),
            prefix + "_resumed_from_checkpoint": bool(resumed)}


def run_store_arm(b, ms, P, ckp, mode="hybrid"):
    """store_query(k = 60) -> 20 pages -> decide n = 48, per mention, checkpointed."""
    ck = json.load(open(ckp)) if os.path.exists(ckp) else {}
    resumed = bool(ck)
    for i, m in enumerate(ms):
        if m["id"] in ck:
            continue
        t = time.perf_counter()
        r = b.call("store_query", q=m["mention"] or m["state"], k=60, mode=mode)
        t_ret = (time.perf_counter() - t) * 1000
        sh = pages_of([h["id"] for h in r.get("hits", [])])
        p = laya_choice(b, m["state"], sh, P, 48, decide=True)
        ck[m["id"]] = {"shortlist": sh, "pred": p, "retrieval_ms": round(t_ret, 1), "total_ms": round((time.perf_counter() - t) * 1000, 1)}
        if i % 20 == 0:
            json.dump(ck, open(ckp + ".tmp", "w")); os.replace(ckp + ".tmp", ckp)
            print(os.path.basename(ckp), i, "/", len(ms), flush=True)
    json.dump(ck, open(ckp + ".tmp", "w")); os.replace(ckp + ".tmp", ckp)
    rows = [ck[m["id"]] for m in ms]
    gold = [m["gold_row"] for m in ms]
    ex = {"shortlists": [r["shortlist"] for r in rows], "latency_ms": [r["total_ms"] for r in rows], "retrieval_ms": [r["retrieval_ms"] for r in rows]}
    ex.update(latency_block(ex["latency_ms"], resumed)); ex.update(latency_block(ex["retrieval_ms"], resumed, "retrieval_ms"))
    return score([r["pred"] for r in rows], [g in r["shortlist"] for r, g in zip(rows, gold)], gold, ex)


def load_pool(b, rows, P, state_file, batch=20000):
    done = set(json.load(open(state_file))) if os.path.exists(state_file) else set()
    t0 = time.time()
    for i in range(0, len(rows), batch):
        key = str(i)
        if key in done:
            continue
        chunk = rows[i:i + batch]
        ab = P.abstracts(chunk)
        b.call("store_ingest_batch", docs=[{"text": "# %s\n\n%s" % (P.titles[r], ab[r]), "source": "pg%d" % r} for r in chunk])
        done.add(key); json.dump(sorted(done), open(state_file + ".tmp", "w")); os.replace(state_file + ".tmp", state_file)
        print("loaded", i + len(chunk), "/", len(rows), round(time.time() - t0, 1), "s", flush=True)
    return round(time.time() - t0, 1)


# ---------------------------------------------------------------- clarification 5 / 5a: the 1M pool

def pool_m1():
    """The 1M pool's sorted rows, checked against the SHA-256 written with it (kilt_el_pool_1m.py)."""
    from kilt_el_pool_1m import rows_sha256
    rows = np.load(os.path.join(EL, "pool_m1.npy"))
    meta = json.load(open(os.path.join(R, "pool_m1.json")))
    sha = rows_sha256(rows)
    assert sha == meta["rows_sha256"] and len(rows) == meta["N"], "pool_m1.npy does not match pool_m1.json"
    assert np.all(np.diff(rows) > 0), "pool rows must be sorted and distinct"
    return rows, sha


def dense_top_in_pool(qv, E, pool, k=20, chunk=250_000):
    """L1 restricted to the pool: exact inner-product top-k of each query among the pool's rows of E only (the pool rows
    are gathered chunk by chunk; no other page is scored). Returns page rows, best first."""
    pool = np.asarray(pool, dtype=np.int64)
    best_s = np.full((len(qv), k), -1e4, dtype=np.float32); best_i = np.full((len(qv), k), -1, dtype=np.int64)
    for s0 in range(0, len(pool), chunk):
        rows = pool[s0:s0 + chunk]
        sc = qv @ np.asarray(E[rows], dtype=np.float32).T
        kk = min(k, sc.shape[1])
        top = np.argpartition(-sc, kk - 1, axis=1)[:, :kk]
        cs = np.concatenate([best_s, np.take_along_axis(sc, top, 1)], 1); ci = np.concatenate([best_i, rows[top]], 1)
        j = np.argsort(-cs, axis=1, kind="stable")[:, :k]
        best_s, best_i = np.take_along_axis(cs, j, 1), np.take_along_axis(ci, j, 1)
    return best_i


def p0_tables(P):
    """bench_el's P0 inputs: the mention -> gold-row counts of AIDA train (the 500 dev-sample mentions excluded), and the
    normalised-title index of all pages."""
    from collections import Counter, defaultdict
    dev_ids = {m["id"] for m in (json.loads(l) for l in open(os.path.join(EL, "mentions_mq.jsonl"))) if m["split"] == "dev"}
    ids = np.load(os.path.join(K, "dense", "ids.npy")); row_of = {str(x): i for i, x in enumerate(ids)}
    prior, n_train = defaultdict(Counter), 0
    for line in open(os.path.join(K, "aidayago2-train-kilt.jsonl")):
        d = json.loads(line)
        if d["id"] in dev_ids or not d["output"] or not d["output"][0].get("provenance"):
            continue
        n_train += 1
        gid = str(d["output"][0]["provenance"][0]["wikipedia_id"])
        if gid in row_of:
            prior[(d["meta"].get("mention") or "").lower().strip()][row_of[gid]] += 1
    title_rows = defaultdict(list)
    for i, t in enumerate(P.titles):
        title_rows[ntitle(t)].append(i)
    return prior, title_rows, n_train


def p0_choose(mention, prior, title_rows, inpool):
    """bench_el's P0, verbatim: the most frequent in-pool gold for the mention string in AIDA train, else the first in-pool
    page whose normalised title equals the normalised mention, else None."""
    from collections import Counter
    men = (mention or "").lower().strip()
    c = [(n, r) for r, n in prior.get(men, Counter()).most_common() if inpool(r)]
    if c:
        return c[0][1]
    tm = [r for r in title_rows.get(ntitle(men), []) if inpool(r)]
    return tm[0] if tm else None


def ns_tables(c):
    """mahabodi_store tables: all of them, and those shared by namespaces (an `ns` column)."""
    tabs = [r[0] for r in c.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'mahabodi_store' ORDER BY tablename").fetchall()]
    shared = {r[0] for r in c.execute("SELECT table_name FROM information_schema.columns WHERE table_schema = 'mahabodi_store' AND column_name = 'ns'").fetchall()}
    return tabs, shared


def store_sizes(c):
    """Per-table sizes of the mahabodi_store schema. total/heap/index bytes are pg_total_relation_size / pg_relation_size /
    pg_indexes_size of the whole table: for a shared table (an `ns` column) that is ALL namespaces together, including
    free space left by deleted rows. Per namespace, shared tables report rows and tuple bytes (sum of pg_column_size of
    the row: row data only, no page overhead, indexes or free space). vec_<ns> tables belong to one namespace."""
    tabs, shared = ns_tables(c)
    out = {}
    for t in tabs:
        q = 'mahabodi_store."%s"' % t
        tot, heap, idx = c.execute("SELECT pg_total_relation_size(%s::regclass), pg_relation_size(%s::regclass), pg_indexes_size(%s::regclass)",
                                   (q, q, q)).fetchone()
        d = {"total_bytes": int(tot), "heap_bytes": int(heap), "index_bytes": int(idx)}
        if t in shared:
            d["per_namespace"] = {ns: {"rows": int(n), "tuple_bytes": int(b or 0)} for ns, n, b in
                                  c.execute("SELECT ns, count(*), sum(pg_column_size(t.*)) FROM %s t GROUP BY ns ORDER BY ns" % q).fetchall()}
        elif t.startswith("vec_"):
            d["namespace"] = t[4:]
            d["rows"] = int(c.execute("SELECT count(*) FROM %s" % q).fetchone()[0])
        out[t] = d
        print("size", t, {k: v for k, v in d.items() if k != "per_namespace"}, flush=True)
    return out


def bytes_per_page(sizes, ns, pages, db_bytes=None):
    """Two figures. (1) all mahabodi_store tables' total size / pages: every namespace in the database counted (the
    prereg's "about 53 KB per page at 4.14M" is this); (2) the namespace's estimated share: its own vec_<ns> table in
    full, plus each shared table's total size times the namespace's share of that table's tuple bytes."""
    total = sum(d["total_bytes"] for d in sizes.values())
    own = 0.0
    for d in sizes.values():
        if "per_namespace" in d:
            tb = sum(v["tuple_bytes"] for v in d["per_namespace"].values())
            mine = d["per_namespace"].get(ns, {}).get("tuple_bytes", 0)
            own += d["total_bytes"] * (mine / tb if tb else 0.0)
        elif d.get("namespace") == ns:
            own += d["total_bytes"]
    tuple_own = sum(d["per_namespace"].get(ns, {}).get("tuple_bytes", 0) for d in sizes.values() if "per_namespace" in d) + \
        sum(d["total_bytes"] for d in sizes.values() if d.get("namespace") == ns)
    return {"pages": pages, "database_bytes": db_bytes, "store_tables_total_bytes_all_namespaces": total,
            "bytes_per_page_all_namespaces": round(total / pages, 1),
            "namespace_estimated_bytes": int(own), "bytes_per_page_namespace_estimated": round(own / pages, 1),
            "namespace_tuple_bytes_plus_own_vector_table": int(tuple_own),
            "bytes_per_page_namespace_tuple_data": round(tuple_own / pages, 1),
            "method": "all_namespaces = sum of pg_total_relation_size over mahabodi_store tables / pages; namespace_estimated = "
                      "vec_<ns> total + sum over shared tables of total size x (ns tuple bytes / all tuple bytes); tuple_data = "
                      "ns tuple bytes (no indexes, page overhead or free space) + vec_<ns> total"}


def parse_load_log(paths, n_all=N_ALL_PAGES):
    """load_pool's progress lines "loaded N / <n_all> S s". S restarts at each process start (resume), so a drop in S
    starts a new segment; the elapsed load time is the sum of each segment's last S (it excludes model loading before
    load_pool and any time between processes)."""
    pat = re.compile(r"^loaded (\d+) / %d ([\d.]+) s\s*$" % n_all)
    segs = []
    for p in paths:
        for line in open(p, errors="replace"):
            m = pat.match(line.strip())
            if not m:
                continue
            n, s = int(m.group(1)), float(m.group(2))
            if not segs or s < segs[-1]["last_s"]:
                segs.append({"file": p, "first_pages": n, "first_s": s, "lines": 0})
            segs[-1].update({"last_pages": n, "last_s": s}); segs[-1]["lines"] += 1
    return {"segments": segs, "elapsed_s_to_last_done_batch": round(sum(x["last_s"] for x in segs), 1),
            "last_pages_logged": segs[-1]["last_pages"] if segs else None, "progress_lines": sum(x["lines"] for x in segs),
            "rule": "a drop in S starts a new segment (a resumed process); elapsed = sum of each segment's last S"}


def load_state_progress(state_file, n_all=N_ALL_PAGES, batch=LOAD_BATCH):
    done = sorted(int(k) for k in json.load(open(state_file)))
    return {"state_file": state_file, "done_batches": len(done), "batch_pages": batch,
            "done_pages": sum(min(batch, n_all - k) for k in done),
            "contiguous_from_0": done == list(range(0, len(done) * batch, batch)),
            "last_batch_start": done[-1] if done else None, "n_all_pages": n_all}


def partial_report(a):
    import psycopg
    if not a.log or not a.reason:
        sys.exit("partial_report needs --log (the load_full log file(s), comma-separated) and --reason")
    c = psycopg.connect(a.dsn, autocommit=True)
    if c.execute("SELECT to_regclass('mahabodi_store.\"vec_full\"')").fetchone()[0] is None:
        sys.exit("mahabodi_store.vec_full does not exist: namespace `full` was already dropped; the partial report must run before drop_full")
    out = {"phase": "partial_report", "namespace": "full", "prereg": "research/PREREG_SCALE_V2.md clarifications 5 and 5a.6",
           "stop_reason": a.reason, "provenance": provenance(), "t0": time.time()}
    out["load_progress"] = load_state_progress(os.path.join(R, "el_store_v2_full.load.json"))
    logs = [p for p in a.log.split(",") if p]
    out["load_log"] = parse_load_log(logs)
    out["load_log"]["files"] = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in logs}
    out["load_log"]["matches_state"] = out["load_log"]["last_pages_logged"] == out["load_progress"]["done_pages"]
    out["database_bytes"] = int(c.execute("SELECT pg_database_size(current_database())").fetchone()[0])
    out["sizes"] = store_sizes(c)
    out["bytes_per_page"] = bytes_per_page(out["sizes"], "full", out["load_progress"]["done_pages"], out["database_bytes"])
    out["seconds_total"] = round(time.time() - out.pop("t0"), 1)
    json.dump(out, open(FULL_PARTIAL, "w"), indent=1, default=str)
    print("PHASE-DONE partial_report", out["load_progress"], {k: v for k, v in out["load_log"].items() if k != "segments"},
          out["bytes_per_page"], flush=True)


def drop_full(a):
    import psycopg
    if not os.path.exists(FULL_PARTIAL):
        sys.exit("refusing: %s does not exist (run --phase partial_report first)" % FULL_PARTIAL)
    rep = json.load(open(FULL_PARTIAL))
    if rep.get("namespace") != "full" or "sizes" not in rep or "vec_full" not in rep["sizes"]:
        sys.exit("refusing: %s is not a complete partial report of namespace full" % FULL_PARTIAL)
    c = psycopg.connect(a.dsn, autocommit=True)
    out = {"phase": "drop_full", "namespace": "full", "partial_report": FULL_PARTIAL, "provenance": provenance(), "t0": time.time(),
           "database_bytes_before": int(c.execute("SELECT pg_database_size(current_database())").fetchone()[0])}
    # Clarification 5b: drop the whole store schema (near-instant, almost no WAL) instead of DELETE + VACUUM of ~220 GB
    # on a disk with ~7.5 GB free. Also removes the small finished namespaces (t100k: bridge; t100kf: D1; devgate:
    # rebuildable for D2 step 2); their results are already recorded. load_1m recreates the schema (store_open).
    out["namespaces_dropped"] = [r[0] for r in c.execute("SELECT ns FROM mahabodi_store.meta ORDER BY ns").fetchall()]
    t = time.time(); c.execute("DROP SCHEMA mahabodi_store CASCADE"); out["drop_schema_s"] = round(time.time() - t, 1)
    c.execute("CHECKPOINT")
    out["database_bytes_after"] = int(c.execute("SELECT pg_database_size(current_database())").fetchone()[0])
    out["note"] = "DROP SCHEMA mahabodi_store CASCADE (clarification 5b); all store namespaces removed"
    out["seconds_total"] = round(time.time() - out.pop("t0"), 1)
    json.dump(out, open(os.path.join(R, "el_store_v2_drop_full.json"), "w"), indent=1, default=str)
    print("PHASE-DONE drop_full", flush=True)


def build_el_m1(a):
    """Clarification 5a.3: a v1-layout database holding exactly the 1M pool's v1 rows, so mahabodi_pg.search runs on it
    unchanged with the pool's own lexical statistics and dense neighbours.
      - schema: deploy/postgres/01_schema.sql and namespace `kilt`, as el_pg_load.ensure_db (search indexes dropped for
        the bulk copy, then rebuilt as el_pg_load.index: GIN on tsv; HNSW vector_cosine_ops m = 16, ef_construction = 64);
      - rows: el_full's `kilt` atf rows whose id is a page passage (pg_<row>_...) of a pool page, all columns but the
        generated tsv (recomputed on insert), and their atf_embedding rows (the page's first-passage KILT vector);
      - vocab (the typo-correction vocabulary search reads): recomputed over the copied rows with el_pg_load.vocab's rule.
    el_full's non-page ATFs (structured ATFs parsed out of page text, which map to no page) are not copied: they cannot
    be attributed to a page, so they cannot be restricted to the pool. They are counted."""
    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    sys.path.insert(0, os.path.join(ROOT, "deploy", "sync"))
    from mahabodi_pg import _terms
    pool, sha = pool_m1()
    db = conninfo_to_dict(a.m1_dsn)["dbname"]
    assert db != V1_DSN_DB and conninfo_to_dict(a.v1_dsn)["dbname"] == V1_DSN_DB, "--m1-dsn must name a new database, --v1-dsn el_full"
    out = {"phase": "build_el_m1", "database": db, "source": V1_DSN_DB, "pool_rows_sha256": sha, "provenance": provenance(),
           "prereg": "research/PREREG_SCALE_V2.md clarification 5a.3", "t0": time.time(), "times": {}}
    admin = psycopg.connect(make_conninfo(a.m1_dsn, dbname="postgres"), autocommit=True)
    if admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,)).fetchone():
        if not a.rebuild:
            sys.exit("database %s exists; pass --rebuild to drop and rebuild it" % db)
        admin.execute('DROP DATABASE "%s" WITH (FORCE)' % db)
    admin.execute('CREATE DATABASE "%s"' % db)
    out["el_full_database_bytes"] = int(admin.execute("SELECT pg_database_size(%s)", (V1_DSN_DB,)).fetchone()[0])
    d = psycopg.connect(a.m1_dsn, autocommit=True)
    d.execute(open(os.path.join(ROOT, "deploy", "postgres", "01_schema.sql")).read())
    d.execute("LOAD 'age'"); d.execute("SET search_path = ag_catalog, mahabodi, public")
    d.execute("SELECT mahabodi.create_namespace('kilt', 'sentence-transformers/all-MiniLM-L6-v2', 384)")
    d.execute("DROP INDEX IF EXISTS mahabodi.atf_tsv_gin"); d.execute("DROP INDEX IF EXISTS mahabodi.atf_embedding_hnsw")
    # the pool's v1 ids, selected inside el_full (session temp tables only; el_full's own tables are only read)
    src = psycopg.connect(a.v1_dsn, autocommit=True)
    t = time.time()
    src.execute("CREATE TEMP TABLE m1_pool (r int PRIMARY KEY)")
    with src.cursor().copy("COPY m1_pool (r) FROM STDIN") as cp:
        for r in pool.tolist():
            cp.write_row((r,))
    src.execute("ANALYZE m1_pool")
    src.execute("CREATE TEMP TABLE v1_ids AS SELECT id, (substring(id from '^pg_([0-9]+)_'))::int AS r FROM mahabodi.atf WHERE namespace = 'kilt'")
    src.execute("CREATE TEMP TABLE m1_ids AS SELECT v.id, v.r FROM v1_ids v JOIN m1_pool p ON p.r = v.r")
    src.execute("CREATE UNIQUE INDEX ON m1_ids (id)"); src.execute("ANALYZE m1_ids")
    v1 = {"atf_rows": src.execute("SELECT count(*) FROM v1_ids").fetchone()[0],
          "non_page_atf_rows_not_copied": src.execute("SELECT count(*) FROM v1_ids WHERE r IS NULL").fetchone()[0],
          "pages_with_rows": src.execute("SELECT count(DISTINCT r) FROM v1_ids WHERE r IS NOT NULL").fetchone()[0],
          "pool_atf_rows": src.execute("SELECT count(*) FROM m1_ids").fetchone()[0],
          "pool_vectors": src.execute("SELECT count(*) FROM mahabodi.atf_embedding e JOIN m1_ids i ON i.id = e.atf_id WHERE e.namespace = 'kilt'").fetchone()[0]}
    src_pages = {r[0] for r in src.execute("SELECT DISTINCT r FROM m1_ids").fetchall()}
    v1["pool_pages_with_rows"] = len(src_pages)
    out["el_full"] = v1; out["times"]["select_s"] = round(time.time() - t, 1)
    print("el_full", v1, flush=True)

    def pipe(sel, ins):
        with src.cursor().copy(sel) as cout, d.cursor().copy(ins) as cin:
            for block in cout:
                cin.write(block)
    cols = "namespace, id, action, input, logic, access, events, data_connections, body, source, content_hash, updated_at"
    t = time.time()
    pipe("COPY (SELECT %s FROM mahabodi.atf a JOIN m1_ids i USING (id) WHERE a.namespace = 'kilt') TO STDOUT" % ", ".join("a." + x for x in cols.split(", ")),
         "COPY mahabodi.atf (%s) FROM STDIN" % cols)
    out["times"]["copy_atf_s"] = round(time.time() - t, 1)
    t = time.time()
    pipe("COPY (SELECT e.namespace, e.atf_id, e.model, e.embedding FROM mahabodi.atf_embedding e JOIN m1_ids i ON i.id = e.atf_id WHERE e.namespace = 'kilt') TO STDOUT",
         "COPY mahabodi.atf_embedding (namespace, atf_id, model, embedding) FROM STDIN")
    out["times"]["copy_embedding_s"] = round(time.time() - t, 1)
    print("copied", out["times"], flush=True)
    # vocabulary: el_pg_load.vocab's rule (terms of id + action + body, document frequency) over the copied rows
    t = time.time(); df = {}
    with d.transaction(), d.cursor(name="v") as cur:
        cur.itersize = 20000
        cur.execute("SELECT id, action, body FROM mahabodi.atf WHERE namespace = 'kilt'")
        for id_, act, bd in cur:
            for w in _terms(" ".join([id_, act, bd])):
                df[w] = df.get(w, 0) + 1
    with d.transaction():
        with d.cursor().copy("COPY mahabodi.vocab (namespace, term, doc_freq) FROM STDIN") as cp:
            for w, n in df.items():
                cp.write_row(("kilt", w, n))
    out["times"]["vocab_s"] = round(time.time() - t, 1); out["vocab_terms"] = len(df)
    # v1's search indexes (el_pg_load.index)
    d.execute("SET maintenance_work_mem = '%s'" % a.v1_mwm); d.execute("SET max_parallel_maintenance_workers = 6")
    out["index_settings"] = {"maintenance_work_mem": a.v1_mwm, "max_parallel_maintenance_workers": 6,
                             "hnsw": "vector_cosine_ops, m = 16, ef_construction = 64 (el_pg_load.index)"}
    for name, sql in (("atf_tsv_gin", "CREATE INDEX atf_tsv_gin ON mahabodi.atf USING gin (tsv)"),
                      ("atf_embedding_hnsw", "CREATE INDEX atf_embedding_hnsw ON mahabodi.atf_embedding "
                                             "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")):
        t = time.time(); d.execute(sql); out["times"][name + "_s"] = round(time.time() - t, 1)
        print("index", name, out["times"][name + "_s"], "s", flush=True)
    t = time.time()
    for tb in ("mahabodi.atf", "mahabodi.atf_embedding", "mahabodi.vocab"):
        d.execute("VACUUM ANALYZE %s" % tb)
    out["times"]["vacuum_analyze_s"] = round(time.time() - t, 1)
    # checks: every pool page with v1 rows is present, with all its rows and its vector
    m1 = {"atf_rows": d.execute("SELECT count(*) FROM mahabodi.atf WHERE namespace = 'kilt'").fetchone()[0],
          "vectors": d.execute("SELECT count(*) FROM mahabodi.atf_embedding WHERE namespace = 'kilt'").fetchone()[0]}
    m1_pages = {r[0] for r in d.execute("SELECT DISTINCT (substring(id from '^pg_([0-9]+)_'))::int FROM mahabodi.atf WHERE namespace = 'kilt'").fetchall()}
    m1["pages"] = len(m1_pages)
    m1["vector_pages"] = d.execute("SELECT count(DISTINCT (substring(atf_id from '^pg_([0-9]+)_'))::int) FROM mahabodi.atf_embedding WHERE namespace = 'kilt'").fetchone()[0]
    pool_set = set(pool.tolist())
    out["el_m1"] = m1
    out["checks"] = {"pool_pages": len(pool_set), "pool_pages_without_v1_rows": sorted(pool_set - src_pages)[:100],
                     "n_pool_pages_without_v1_rows": len(pool_set - src_pages),
                     "pages_equal": m1_pages == src_pages, "pages_outside_pool": len(m1_pages - pool_set),
                     "atf_rows_equal": m1["atf_rows"] == v1["pool_atf_rows"], "vectors_equal": m1["vectors"] == v1["pool_vectors"],
                     "one_vector_per_page": m1["vectors"] == m1["vector_pages"] == m1["pages"]}
    out["checks"]["ok"] = all(out["checks"][k] for k in ("pages_equal", "atf_rows_equal", "vectors_equal")) and not out["checks"]["pages_outside_pool"]
    # the pool pages absent from el_full: who they are, why (their KILT text), and whether any is a mention's gold
    missing = sorted(pool_set - src_pages)
    golds = {m["gold_row"] for w in ("fresh", "test", "dev") for m in load_mentions(w)}
    P = Pages(); mab = P.abstracts(missing)
    out["checks"]["missing_pages_detail"] = {
        "ids": missing, "gold_overlap": sorted(set(missing) & golds),
        "kilt_source": [{"r": r, "title": P.titles[r], "abstract_chars": len(mab[r] or "")} for r in missing],
        "note": "pool pages with no atf row in el_full (the v1 load produced no passage for them); el_full holds rows "
                "for %d of %d pages overall" % (v1["pages_with_rows"], N_ALL_PAGES)}
    # sizes: pg_total_relation_size of the relation plus any declarative partitions (pg_partition_tree returns no
    # rows for a plain table on this server, which crashed the first run: int(None))
    def relsize(rel, fn):
        kids = [r[0] for r in d.execute("SELECT inhrelid::regclass::text FROM pg_inherits WHERE inhparent = %s::regclass", (rel,)).fetchall()]
        return sum(int(d.execute("SELECT " + fn + "(%s::regclass)", (x,)).fetchone()[0]) for x in [rel] + kids)
    sz = {}
    try:
        for rel in ("mahabodi.atf", "mahabodi.atf_embedding", "mahabodi.vocab"):
            sz[rel] = {"total_bytes": relsize(rel, "pg_total_relation_size")}
        for idx in ("mahabodi.atf_tsv_gin", "mahabodi.atf_embedding_hnsw", "mahabodi.vocab_term_trgm"):
            sz[idx] = {"index_bytes": relsize(idx, "pg_relation_size")}
    except Exception as e:  # noqa: BLE001  a sizing failure must not lose the checks
        sz["error"] = str(e)
    out["sizes"] = sz
    out["database_bytes"] = int(d.execute("SELECT pg_database_size(current_database())").fetchone()[0])
    out["bytes_per_page"] = round(out["database_bytes"] / max(1, m1["pages"]), 1)
    out["seconds_total"] = round(time.time() - out.pop("t0"), 1)
    json.dump(out, open(os.path.join(R, "el_store_v2_build_el_m1.json"), "w"), indent=1, default=str)
    print("checks", out["checks"], flush=True)
    if not out["checks"]["ok"]:
        sys.exit("BUILD-EL-M1-CHECK-FAILED")
    print("PHASE-DONE build_el_m1", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["bridge", "load_full", "partial_report", "drop_full", "load_1m", "build_el_m1",
                                                       "probes", "primary", "v1check", "ablation"])
    ap.add_argument("--dsn", default=None, help="the store database (namespaces t100k, full, m1); every phase but build_el_m1")
    ap.add_argument("--v1-dsn", default="host=127.0.0.1 port=5433 user=postgres dbname=el_full")
    ap.add_argument("--m1-dsn", default="host=127.0.0.1 port=5433 user=postgres dbname=el_m1", help="the v1-layout 1M-pool database (build_el_m1, ablation)")
    ap.add_argument("--namespace", default="m1", help="probes: the store namespace whose vectors and IVFFlat index are swept")
    ap.add_argument("--log", default=None, help="partial_report: the load_full log file(s), comma-separated, in order")
    ap.add_argument("--reason", default=None, help="partial_report: why load_full stopped")
    ap.add_argument("--rebuild", action="store_true", help="build_el_m1: drop and rebuild an existing el_m1")
    ap.add_argument("--v1-mwm", default="4GB", help="build_el_m1: maintenance_work_mem for v1's index builds (el_pg_load's default)")
    ap.add_argument("--laya", default=os.path.join(ROOT, "models", "laya-v2"))
    ap.add_argument("--vector-type", default=None, help="halfvec or vector (default: from store_vector_index_dev100k.json)")
    a = ap.parse_args()
    if a.phase != "build_el_m1" and not a.dsn:
        ap.error("--dsn is required for --phase %s" % a.phase)
    if a.phase in ("partial_report", "drop_full", "build_el_m1"):  # database-only phases: no models, no page table
        {"partial_report": partial_report, "drop_full": drop_full, "build_el_m1": build_el_m1}[a.phase](a)
        return
    from mahabodi import Bodi
    step2 = json.load(open(os.path.join(R, "store_vector_index_dev100k.json")))
    vtype = a.vector_type or ("halfvec" if step2["halfvec_use_at_5_9M"] else "vector")
    P = Pages()
    b = Bodi(); b.load_laya(a.laya, intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    no_decision_cache(b)
    out = {"phase": a.phase, "decide_cache": False, "provenance": provenance(), "prereg": "research/PREREG_SCALE_V2.md", "vector_type": vtype, "t0": time.time()}
    fname = os.path.join(R, "el_store_v2_%s.json" % a.phase)
    if a.phase == "probes":
        out["namespace"] = a.namespace
        if a.namespace != "m1":  # el_store_v2_probes.json is the m1 setting that primary and ablation read
            fname = os.path.join(R, "el_store_v2_probes_%s.json" % a.namespace)

    if a.phase == "bridge":
        ms = load_mentions("test"); gold = [m["gold_row"] for m in ms]
        el = json.load(open(os.path.join(R, "bench_el.json")))
        assert el["mention_ids"] == [m["id"] for m in ms]
        ref = el["stages"]["100000"]["arms"]["M"]
        b.call("store_open", dsn=a.dsn, namespace="t100k", vector_type=vtype)
        pool = np.load(os.path.join(EL, "pool_test_100000_mq.npy")).tolist()
        st = os.path.join(R, "el_store_v2_t100k.load.json")
        times = {"load_s": load_pool(b, pool, P, st)}
        t = time.time(); b.call("store_build_index"); times["build_s"] = round(time.time() - t, 1)
        t = time.time(); times["density"] = b.call("store_ensure_density"); times["density_s"] = round(time.time() - t, 1)
        exact = run_store_arm(b, ms, P, os.path.join(R, "el_store_v2_bridge_exact.ckpt.json"))
        rows = b.call("store_stats")["passages"]
        probes = step2["ivfflat"]["selected_probes_for_bridge"]
        t = time.time(); times["ivfflat_build"] = b.call("store_build_ivfflat_index", lists=math.ceil(math.sqrt(rows)), workers=6, maintenance_mem="10GB", probes=probes)
        times["ivfflat_build_s"] = round(time.time() - t, 1); times["store_stats"] = b.call("store_stats")  # the index definitions in place
        import psycopg
        def idx_scans():  # the store's dense queries must use the IVFFlat index (store/pg.rs dense_tx), not an exact scan
            with psycopg.connect(a.dsn, autocommit=True) as sc:
                sc.execute("SELECT pg_stat_force_next_flush()")
                return sc.execute("SELECT coalesce(sum(idx_scan), 0) FROM pg_stat_user_indexes WHERE indexrelname = 'vec_t100k_ivfflat'").fetchone()[0]
        scans0 = idx_scans()
        ivf = run_store_arm(b, ms, P, os.path.join(R, "el_store_v2_bridge_ivf.ckpt.json"))
        time.sleep(11)  # idle backends flush their statistics within 10 s
        times["ivfflat_idx_scans_during_arm"] = int(idx_scans() - scans0)  # < 1000 only if the arm resumed from a checkpoint
        assert times["ivfflat_idx_scans_during_arm"] > 0, "the IVFFlat arm never scanned the index: %s" % times
        out.update({"mention_ids": [m["id"] for m in ms], "gold_rows": gold, "times": times, "probes": probes,
                    "in_process_M": {"accuracy": ref["accuracy"]}, "store_exact": exact, "store_ivfflat": ivf,
                    "mcnemar_exact_vs_in_process": mcnemar([p == g for p, g in zip(exact["pred"], gold)], [p == g for p, g in zip(ref["pred"], gold)]),
                    "mcnemar_ivf_vs_in_process": mcnemar([p == g for p, g in zip(ivf["pred"], gold)], [p == g for p, g in zip(ref["pred"], gold)])})

    elif a.phase == "load_full":
        b.call("store_open", dsn=a.dsn, namespace="full", vector_type=vtype)
        n_pages = len(P.titles)
        times = {"load_s": load_pool(b, list(range(n_pages)), P, os.path.join(R, "el_store_v2_full.load.json"))}
        t = time.time(); b.call("store_build_index"); times["build_s"] = round(time.time() - t, 1)
        t = time.time(); times["density"] = b.call("store_ensure_density"); times["density_s"] = round(time.time() - t, 1)
        rows = b.call("store_stats")["passages"]
        t = time.time(); times["ivfflat_build"] = b.call("store_build_ivfflat_index", lists=math.ceil(math.sqrt(rows)), workers=6, maintenance_mem="10GB", probes=10)
        times["ivfflat_build_s"] = round(time.time() - t, 1)
        out.update({"pages": n_pages, "passages": rows, "times": times, "stats": b.call("store_stats")})

    elif a.phase == "load_1m":
        # clarification 5: load_full on the 1M constructed pool, namespace m1 (one segment; resumable by batch)
        import psycopg
        pool, sha = pool_m1()
        b.call("store_open", dsn=a.dsn, namespace="m1", vector_type=vtype)
        st = os.path.join(R, "el_store_v2_m1.load.json")
        resumed = os.path.exists(st)  # load_s then covers only the batches loaded by this process
        times = {"load_s": load_pool(b, pool.tolist(), P, st), "load_resumed_from_state": resumed}
        t = time.time(); b.call("store_build_index"); times["build_s"] = round(time.time() - t, 1)
        t = time.time(); times["density"] = b.call("store_ensure_density"); times["density_s"] = round(time.time() - t, 1)
        rows = b.call("store_stats")["passages"]
        lists = math.ceil(math.sqrt(rows))
        t = time.time(); times["ivfflat_build"] = b.call("store_build_ivfflat_index", lists=lists, workers=6, maintenance_mem="10GB", probes=10)
        times["ivfflat_build_s"] = round(time.time() - t, 1)
        with psycopg.connect(a.dsn, autocommit=True) as c:
            db_bytes = int(c.execute("SELECT pg_database_size(current_database())").fetchone()[0])
            sizes = store_sizes(c)
        out.update({"namespace": "m1", "pool_rows_sha256": sha, "pages": len(pool), "passages": rows,
                    "ivfflat": {"lists": lists, "maintenance_work_mem": "10GB", "workers": 6}, "times": times,
                    "stats": b.call("store_stats"), "database_bytes": db_bytes, "sizes": sizes,
                    "bytes_per_page": bytes_per_page(sizes, "m1", len(pool), db_bytes),
                    "label": "1M-page constructed pool (golds + mined hard negatives + random fill)",
                    "size_note": "shared tables hold every namespace in the store database and the free space left by drop_full's "
                                 "delete; the per-namespace tuple bytes and the vec_m1 table are the m1-only figures"})

    elif a.phase == "probes":
        import psycopg
        ns = a.namespace
        assert re.fullmatch(r"[A-Za-z0-9_]+", ns), ns
        vt = 'mahabodi_store."vec_%s"' % ns
        ms = load_mentions("dev")
        qs = np.asarray(b.embed_text([m["mention"] or m["state"] for m in ms]), dtype=np.float32)
        c = psycopg.connect(a.dsn, autocommit=True)
        # exact top-50 by streaming all of the namespace's stored passage vectors (NumPy), in chunks
        best_s = np.full((len(qs), 50), -1e9, dtype=np.float32); best_i = np.full((len(qs), 50), "", dtype=object)
        n_vec = 0
        with c.transaction(), c.cursor(name="v") as cur:  # a server-side cursor needs a transaction on an autocommit connection
            cur.itersize = 200_000
            cur.execute("SELECT node_id, embedding::text FROM %s" % vt)
            while True:
                chunk = cur.fetchmany(200_000)
                if not chunk:
                    break
                ids = np.array([r[0] for r in chunk], dtype=object); n_vec += len(chunk)
                V = np.array([[float(x) for x in r[1][1:-1].split(",")] for r in chunk], dtype=np.float32)
                sc = qs @ V.T
                cs = np.concatenate([best_s, sc], 1)
                ci = np.concatenate([best_i, np.broadcast_to(ids, sc.shape)], 1)
                j = np.argsort(-cs, axis=1, kind="stable")[:, :50]
                best_s, best_i = np.take_along_axis(cs, j, 1), np.take_along_axis(ci, j, 1)
        exact = [set(r) for r in best_i]
        sweep = {}
        c.execute("SET enable_seqscan = off")  # each row must measure the index (see store_vector_index.py); plans recorded
        for p in (10, 20, 40, 80, 160, 320):
            c.execute("SET ivfflat.probes = %d" % p)
            plan = "\n".join(r[0] for r in c.execute("EXPLAIN SELECT node_id FROM %s ORDER BY embedding <#> %%s::text::%s LIMIT 50" % (vt, vtype),
                                                      ("[" + ",".join("%.7f" % x for x in qs[0]) + "]",)).fetchall())
            assert "Index Scan using vec_%s_ivfflat" % ns in plan, plan
            rec, lat = [], []
            for q, ex in zip(qs, exact):
                t = time.perf_counter()
                got = [r[0] for r in c.execute("SELECT node_id FROM %s ORDER BY embedding <#> %%s::text::%s LIMIT 50" % (vt, vtype),
                                               ("[" + ",".join("%.7f" % x for x in q) + "]",)).fetchall()]
                lat.append((time.perf_counter() - t) * 1000); rec.append(len(set(got) & ex) / 50)
            sweep[str(p)] = {"recall_at_50": round(float(np.mean(rec)), 4), "latency_ms_p50": round(float(np.median(lat)), 1),
                             "latency_ms_p95": round(float(np.percentile(lat, 95)), 1), "plan": plan}
            print("probes", p, {k: x for k, x in sweep[str(p)].items() if k != "plan"}, flush=True)
        ok = [int(p) for p, r in sweep.items() if r["recall_at_50"] >= 0.98]
        out.update({"mention_ids": [m["id"] for m in ms], "vectors_scanned_exact": n_vec, "sweep": sweep,
                    "selected_probes": min(ok) if ok else 320, "reached_0_98": bool(ok)})

    elif a.phase == "primary":
        chk = extractor_self_check()
        assert chk["ok"], "v2 extractor does not reproduce mentions_mq.jsonl: %s" % chk
        out["extractor_self_check"] = chk
        pr = json.load(open(os.path.join(R, "el_store_v2_probes.json")))
        assert pr.get("namespace") == "m1", "el_store_v2_probes.json is not the m1 sweep"
        probes = pr["selected_probes"]
        pool, sha = pool_m1()
        pool_set = set(pool.tolist()); inpool = pool_set.__contains__
        b.call("store_open", dsn=a.dsn, namespace="m1", vector_type=vtype, create=False)
        b.call("store_set_probes", probes=probes)
        E = np.load(os.path.join(K, "dense", "emb.f16.npy"), mmap_mode="r")
        prior, title_rows, n_train = p0_tables(P)
        import psycopg

        def m1_idx_scans():
            with psycopg.connect(a.dsn, autocommit=True) as sc:
                sc.execute("SELECT pg_stat_force_next_flush()")
                return int(sc.execute("SELECT coalesce(sum(idx_scan), 0) FROM pg_stat_user_indexes WHERE indexrelname = 'vec_m1_ivfflat'").fetchone()[0])
        scans0 = m1_idx_scans()
        res = {}
        for label, ms in (("fresh", load_mentions("fresh")), ("v1_items", load_mentions("test"))):
            gold = [m["gold_row"] for m in ms]
            assert all(inpool(g) for g in gold), "a gold page is missing from the 1M pool"
            store = run_store_arm(b, ms, P, os.path.join(R, "el_store_v2_primary_m1_%s.ckpt.json" % label))
            # L1 restricted to the pool: Laya on the dense top-20 among the 1M pool pages only (mention query), n = 16
            t_dense = time.time()
            qv = np.asarray(b.embed_text([m["mention"] or m["state"] for m in ms]), dtype=np.float32)
            best_i = dense_top_in_pool(qv, E, pool, k=20)
            dense_batch_s = time.time() - t_dense
            l1p, l1lat = [], []
            for m, row in zip(ms, best_i):
                t = time.perf_counter(); l1p.append(laya_choice(b, m["state"], row.tolist(), P, 16, decide=False)); l1lat.append((time.perf_counter() - t) * 1000)
            # L1's retrieval is one batched exact NumPy scan for all queries (not per query): reported as a batch total
            # and a per-query mean, next to the per-query decide latency (clarification 3d)
            l1ex = {"decide_ms": [round(x, 1) for x in l1lat], "dense_batch_s": round(dense_batch_s, 1),
                    "dense_batch_ms_per_query": round(dense_batch_s * 1000 / len(ms), 1)}
            l1ex.update(latency_block(l1lat, False, "decide_ms"))
            l1 = score(l1p, [g in row.tolist() for g, row in zip(gold, best_i)], gold, l1ex)
            # P0 (5a.2): bench_el's context-free control within the pool
            p0p = [p0_choose(m.get("mention"), prior, title_rows, inpool) for m in ms]
            c0 = [p == g for p, g in zip(p0p, gold)]
            p0 = {"accuracy": round(float(np.mean(c0)), 4), "ci95": wilson(sum(c0), len(c0)), "answered": sum(p is not None for p in p0p), "pred": p0p}
            cs_, cl_ = [p == g for p, g in zip(store["pred"], gold)], [p == g for p, g in zip(l1p, gold)]
            res[label] = {"mention_ids": [m["id"] for m in ms], "gold_rows": gold, "store_M": store, "L1": l1, "P0": p0,
                          "mcnemar_store_vs_L1": mcnemar(cs_, cl_), "mcnemar_P0_vs_L1": mcnemar(c0, cl_), "mcnemar_store_vs_P0": mcnemar(cs_, c0),
                          "pool_biased": p0["accuracy"] >= l1["accuracy"]}
            print(label, "store", store["accuracy"], "L1", l1["accuracy"], "P0", p0["accuracy"], res[label]["mcnemar_store_vs_L1"], flush=True)
        time.sleep(11)  # idle backends flush their statistics within 10 s
        out["ivfflat_idx_scans_during_store_arms"] = m1_idx_scans() - scans0
        assert out["ivfflat_idx_scans_during_store_arms"] > 0, "store M never scanned vec_m1_ivfflat (or both arms resumed fully from checkpoints)"
        # Clarification 5c: M_exact — the same store arm with the IVFFlat index dropped, so dense retrieval is an exact
        # full scan (store/pg.rs: without a vector index dense search is exact). Fresh items only; descriptive, never
        # the headline; a different configuration, so its latency is labelled and not comparable to store_M's.
        with psycopg.connect(a.dsn, autocommit=True) as sc:
            sc.execute('DROP INDEX mahabodi_store."vec_m1_ivfflat"')
            assert sc.execute("SELECT to_regclass(%s)", ('mahabodi_store."vec_m1_ivfflat"',)).fetchone()[0] is None
        ms_f = load_mentions("fresh"); gold_f = [m["gold_row"] for m in ms_f]
        ex_arm = run_store_arm(b, ms_f, P, os.path.join(R, "el_store_v2_primary_m1_fresh_exact.ckpt.json"))
        ce_f = [p == g for p, g in zip(ex_arm["pred"], gold_f)]
        cs_f = [p == g for p, g in zip(res["fresh"]["store_M"]["pred"], gold_f)]
        res["fresh"]["store_M_exact"] = ex_arm
        res["fresh"]["mcnemar_store_vs_store_exact"] = mcnemar(cs_f, ce_f)
        print("fresh store_M_exact", ex_arm["accuracy"], "vs store_M", res["fresh"]["store_M"]["accuracy"],
              res["fresh"]["mcnemar_store_vs_store_exact"], flush=True)
        t = time.time()
        rebuilt = b.call("store_build_ivfflat_index", lists=math.ceil(math.sqrt(b.call("store_stats")["passages"])),
                         workers=6, maintenance_mem="10GB", probes=probes)
        out["m_exact"] = {"prereg": "clarification 5c (descriptive; isolates the ANN approximation's cost to store M at 1M)",
                          "index_dropped_before": True, "ivfflat_rebuilt_after": rebuilt, "ivfflat_rebuild_s": round(time.time() - t, 1)}
        out.update({"namespace": "m1", "pool_rows_sha256": sha, "pool_pages": len(pool), "probes": probes, "results": res,
                    "L1_retrieval": "exact inner product over the pool's rows of dense/emb.f16.npy only (top-20 among the 1M pool pages)",
                    "P0_source": "bench_el's P0: AIDA-train mention -> gold counts (the 500 dev-sample mentions excluded; %d training "
                                 "mentions), most frequent in-pool gold, else the first in-pool page with the normalised title" % n_train,
                    "pool_biased_rule": "P0 accuracy >= L1 accuracy (bench_el)",
                    "label": "1M-page constructed pool (golds + mined hard negatives + random fill); adversarial to L1's retriever "
                             "by construction (hard negatives mined from the same dense index and BM25)"})

    elif a.phase in ("v1check", "ablation"):
        sys.path.insert(0, os.path.join(ROOT, "deploy", "sync"))
        import mahabodi_pg
        # v1check: el_full (5.9M), unchanged. ablation: the v1 corner rebuilt on the 1M pool (el_m1, clarification 5a.3)
        v1 = mahabodi_pg.connect(a.v1_dsn if a.phase == "v1check" else a.m1_dsn); v1.execute("SET hnsw.ef_search = 100")
        if a.phase == "v1check":
            ms = load_mentions("test"); gold = [m["gold_row"] for m in ms]
            pg1 = json.load(open(os.path.join(R, "bench_el_pg.json")))
            assert pg1["mention_ids"] == [m["id"] for m in ms]
            # (1) mahabodi_pg.search itself; (2) the harness composition the ablation's v1 corner uses (v1 lexical list +
            # v1 first-passage vectors, fused by rrf()). Both must reproduce v1's per-item predictions.
            preds, preds_h, same_sh = [], [], 0
            for m in ms:
                q = m["mention"] or m["state"]; qv = b.embed_text([q])[0]
                hits = mahabodi_pg.search(v1, "kilt", q, qvec=qv, k=60, model="minilm")
                sh = pages_of([h["id"] for h in hits])
                pgtext = [h["id"] for h in mahabodi_pg.search(v1, "kilt", q, qvec=None, k=60, model="minilm")]
                vec = "[" + ",".join("%.7g" % x for x in qv) + "]"
                fdense = [r[0] for r in v1.execute("SELECT atf_id FROM mahabodi.atf_embedding WHERE namespace = 'kilt' AND model = 'minilm' "
                                                   "ORDER BY embedding <=> %s::vector, atf_id LIMIT 50", (vec,)).fetchall()]
                sh_h = pages_of(rrf(pgtext, fdense)[:60])
                same_sh += sh == sh_h
                preds.append(laya_choice(b, m["state"], sh, P, 48, decide=True))
                preds_h.append(laya_choice(b, m["state"], sh_h, P, 48, decide=True))
            ref = pg1["dbs"]["el_full"]["arms"]["M_pg"]["pred"]
            agree = sum(x == y for x, y in zip(preds, ref)); agree_h = sum(x == y for x, y in zip(preds_h, ref))
            out.update({"mention_ids": [m["id"] for m in ms], "pred_search": preds, "pred_harness": preds_h,
                        "agree_search_with_bench_el_pg": agree, "agree_harness_with_bench_el_pg": agree_h, "same_shortlist_search_vs_harness": same_sh,
                        "equal": agree == len(ms) and agree_h == len(ms)})
        else:
            assert json.load(open(os.path.join(R, "el_store_v2_v1check.json")))["equal"], "v1 corner does not reproduce v1: fix or disclose first"
            pr = json.load(open(os.path.join(R, "el_store_v2_probes.json")))
            assert pr.get("namespace") == "m1", "el_store_v2_probes.json is not the m1 sweep"
            probes = pr["selected_probes"]
            pool, sha = pool_m1()
            bm = json.load(open(os.path.join(R, "el_store_v2_build_el_m1.json")))
            assert bm["checks"]["ok"] and bm["pool_rows_sha256"] == sha, "el_m1 was not built from this pool, or its checks failed"
            out.update({"namespace": "m1", "pool_rows_sha256": sha, "v1_corner_database": bm["database"]})
            b.call("store_open", dsn=a.dsn, namespace="m1", vector_type=vtype, create=False)
            b.call("store_set_probes", probes=probes)
            out["extractor_self_check"] = extractor_self_check()
            assert out["extractor_self_check"]["ok"]
            ms = load_mentions("fresh"); gold = [m["gold_row"] for m in ms]
            cells = {k: {"pred": [], "hit": []} for k in ("passage_cascade", "passage_pgtext", "first_cascade", "first_pgtext")}
            for i, m in enumerate(ms):
                q = m["mention"] or m["state"]; qv = b.embed_text([q])[0]
                casc = [h["id"][2:] for h in b.call("store_query", q=q, k=60, mode="lexical").get("hits", [])]
                pdense = [h["id"][2:] for h in b.call("store_query", q=q, k=50, mode="dense").get("hits", [])]
                pgtext = [h["id"] for h in mahabodi_pg.search(v1, "kilt", q, qvec=None, k=60, model="minilm")]
                vec = "[" + ",".join("%.7g" % x for x in qv) + "]"
                fdense = [r[0] for r in v1.execute("SELECT atf_id FROM mahabodi.atf_embedding WHERE namespace = 'kilt' AND model = 'minilm' "
                                                   "ORDER BY embedding <=> %s::vector, atf_id LIMIT 50", (vec,)).fetchall()]
                lists = {"passage_cascade": rrf(casc, pdense), "passage_pgtext": rrf(pgtext, pdense),
                         "first_cascade": rrf(casc, fdense), "first_pgtext": rrf(pgtext, fdense)}
                for k_, ids in lists.items():
                    sh = pages_of(ids[:60])
                    cells[k_]["pred"].append(laya_choice(b, m["state"], sh, P, 48, decide=True)); cells[k_]["hit"].append(m["gold_row"] in sh)
                if i % 50 == 0:
                    print("ablation", i, "/", len(ms), flush=True)
            out.update({"mention_ids": [m["id"] for m in ms], "gold_rows": gold,
                        "cells": {k: score(v["pred"], v["hit"], gold) for k, v in cells.items()},
                        "note": "descriptive; on the 1M constructed pool (store namespace m1). first_pgtext is the v1 retriever with harness "
                                "RRF, rebuilt on the pool's v1 rows (el_m1: its own lexical statistics, vocabulary and HNSW); v1check "
                                "shows on el_full at 5.9M that mahabodi_pg.search and this composition reproduce v1"})
    out["seconds_total"] = round(time.time() - out.pop("t0"), 1)
    json.dump(out, open(fname, "w"), indent=1, default=str)
    print("PHASE-DONE", a.phase, flush=True)


if __name__ == "__main__":
    main()
