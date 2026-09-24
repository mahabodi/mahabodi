"""End-to-end test of the enterprise store pattern (Enterprise.md, patterns P2-P4).

  1. build memory in process with MahaBodi (fastmemory fixtures, or N SQuAD paragraphs)
  2. embed every ATF with MahaBodi's MiniLM embedder
  3. load snapshot + embeddings into PostgreSQL (AGE graph + pgvector + FTS + trigram vocab)
  4. hybrid search in SQL, including a misspelled query
  5. Cypher neighbourhood of the hits -> hydrate a working set -> Bodi.restore() in a fresh engine
  6. the hydrated engine must answer the same query from the working set (matched, no handoff)
  7. measure: embedding throughput, load time, storage bytes per ATF, SQL query latency

Needs a PostgreSQL with age, vector and pg_trgm available (see test_store.sh), MahaBodi's
Python binding, models/minilm, and ORT_DYLIB_PATH pointing at libonnxruntime.

    python deploy/postgres/test_store.py --dsn "host=127.0.0.1 port=55432 user=me dbname=mb_test" [--paragraphs-file f.jsonl]
"""
import argparse, json, os, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "deploy", "sync"))
import mahabodi_pg as mpg  # noqa: E402
from mahabodi import Bodi  # noqa: E402

FIXTURES = ["business_analytics", "email_analysis", "health_science", "robotics", "world_events"]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--ns", default="e2e")
    ap.add_argument("--paragraphs-file", help="JSONL of {\"text\": ...}; default: one fastmemory fixture")
    ap.add_argument("--fixture", default="robotics", choices=FIXTURES)
    ap.add_argument("--out")
    ap.add_argument("--reuse", action="store_true", help="namespace already loaded with this corpus: skip steps 2-3 (embedding + load)")
    a = ap.parse_args()
    res, fails = {}, []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)

    # 1. in-process memory
    b = Bodi()
    if a.paragraphs_file:
        docs = [{"text": json.loads(l)["text"], "format": "text", "source": "p%d" % i} for i, l in enumerate(open(a.paragraphs_file))]
    else:
        # the fixtures all reuse ids ATF_S_0..19 (ids are unique per source, not globally), so one
        # fixture = one memory; namespaces are how the store keeps such sources apart
        docs = [{"text": open(os.path.join(ROOT, "crates/mahabodi-core/tests/fixtures/%s.md" % a.fixture)).read(), "source": a.fixture}]
    t0 = time.time(); b.ingest_batch(docs); res["inprocess_ingest_s"] = round(time.time() - t0, 2)
    snap = b.snapshot()
    ids = [x["id"] for x in snap["atfs"]]
    res["atfs"] = len(ids)
    check(len(ids) > 0, "in-process ingest produced %d ATFs" % len(ids))

    # 2. embeddings (MahaBodi's own MiniLM, same vectors its hybrid stage uses)
    b.load_embedder(os.path.join(ROOT, "models", "minilm"))
    embed_ids = ids if not a.reuse else ids[:256]   # --reuse: time a 256-passage sample only
    t0 = time.time()
    vecs = {}
    for i in range(0, len(embed_ids), 64):
        chunk = embed_ids[i:i + 64]
        for k, v in zip(chunk, b.embed_text([snap["texts"].get(x, x) for x in chunk])):
            vecs[k] = v
    el = time.time() - t0
    res["embed_passages_per_s"] = round(len(embed_ids) / el, 1)
    res["embed_sample"] = len(embed_ids)
    check(len(vecs) == len(embed_ids) and len(next(iter(vecs.values()))) == 384, "embedded %d ATFs (384-d)" % len(vecs))

    # 3. load into PostgreSQL + AGE + pgvector
    c = mpg.connect(a.dsn)
    if a.reuse:
        t0 = time.time(); mpg.refresh_degrees(c, a.ns); res["refresh_degrees_s"] = round(time.time() - t0, 2)
        res["pg_load_s"] = "not measured (--reuse)"
    else:
        load_fresh(c, a, snap, vecs, res)
    n_pg = c.execute("SELECT count(*) FROM mahabodi.atf WHERE namespace = %s", (a.ns,)).fetchone()[0]
    n_fn = json.loads(mpg._cypher(c, "mb_" + a.ns, "MATCH (f:Function) RETURN count(f)", cols="n agtype")[0][0])
    check(n_pg == len(ids) and n_fn == len(ids), "store holds %d ATF rows and %d Function vertices" % (n_pg, n_fn))
    _search_and_hydrate(a, b, c, snap, ids, res, check)
    _storage(c, a, snap, ids, res)
    res["passed"] = not fails
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(1 if fails else 0)


def load_fresh(c, a, snap, vecs, res):
    c.execute("DELETE FROM mahabodi.namespace WHERE name = %s", (a.ns,))
    if c.execute("SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", ("mb_" + a.ns,)).fetchone():
        c.execute("SELECT ag_catalog.drop_graph(%s::name, true)", ("mb_" + a.ns,))
    for t in ("atf", "vocab"):
        c.execute("DELETE FROM mahabodi.%s WHERE namespace = %%s" % t, (a.ns,))
    c.execute("SELECT mahabodi.create_namespace(%s, %s, 384)", (a.ns, MODEL))
    t0 = time.time()
    res["pg_load"] = mpg.load_snapshot(c, a.ns, snap, embeddings=vecs, model=MODEL)
    res["pg_load_s"] = round(time.time() - t0, 2)


def _search_and_hydrate(a, b, c, snap, ids, res, check):
    # 4-6. search -> neighbourhood -> hydrate -> in-process answer
    probes = []
    for x in snap["atfs"][:: max(1, len(ids) // 20)][:20]:
        words = [w for w in mpg._terms(snap["texts"].get(x["id"], "")) if len(w) >= 6]
        if len(words) >= 2:
            probes.append((x["id"], " ".join(sorted(words)[:3])))
    lat, nlat, hit5, hyd_ok, wsize = [], [], 0, 0, []
    for gold, q in probes:
        qv = b.embed_text([q])[0]
        t0 = time.perf_counter(); hits = mpg.search(c, a.ns, q, qvec=qv, k=5, model=MODEL); lat.append((time.perf_counter() - t0) * 1000)
        top = [h["id"] for h in hits]
        hit5 += gold in top
        t0 = time.perf_counter(); nb = mpg.neighbourhood(c, a.ns, top[:3], limit=100); nlat.append((time.perf_counter() - t0) * 1000)
        work = sorted(set(top) | set(nb)); wsize.append(len(work))
        w = Bodi(); w.restore(mpg.hydrate(c, a.ns, work))
        r = w.query(q, k=5)
        hyd_ok += bool(r["matched"] and not r["handoff"] and any(h["id"][2:] in work for h in r["hits"]))
    res["probe_queries"] = len(probes)
    res["sql_search_recall_at5"] = round(hit5 / max(1, len(probes)), 3)
    res["sql_search_ms_p50"] = round(statistics.median(lat), 2) if lat else None
    res["cypher_neighbourhood_ms_p50"] = round(statistics.median(nlat), 2) if nlat else None
    res["working_set_atfs_median"] = statistics.median(wsize) if wsize else None
    res["hydrated_engine_answered"] = hyd_ok
    check(len(probes) >= 5 and hit5 == len(probes), "SQL hybrid search finds the source ATF in top-5 for %d/%d probes" % (hit5, len(probes)))
    check(hyd_ok == len(probes), "hydrated in-process engine answered %d/%d probes from its working set" % (hyd_ok, len(probes)))
    # typo tolerance through the trigram vocabulary
    gold, q = probes[0]
    word = max(q.split(), key=len)
    typo = q.replace(word, word[:2] + word[3:], 1)
    hits = mpg.search(c, a.ns, typo, qvec=b.embed_text([typo])[0], k=5, model=MODEL)
    res["typo_example"] = {"query": typo, "corrected_terms": hits[0]["corrected_terms"] if hits else [], "found": gold in [h["id"] for h in hits]}
    check(bool(hits) and word in hits[0]["corrected_terms"], "typo %r corrected to %r via pg_trgm" % (typo, word))


def _storage(c, a, snap, ids, res):
    # 7. storage for THIS namespace: its atf/embedding/vocab rows (estimated from row share of the
    # shared tables) plus its own AGE graph schema
    share = c.execute("SELECT (SELECT count(*) FROM mahabodi.atf WHERE namespace = %s)::float8 / greatest(1, (SELECT count(*) FROM mahabodi.atf))", (a.ns,)).fetchone()[0]
    rel = float(c.execute("""SELECT coalesce(sum(pg_total_relation_size(c.oid)), 0) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                       WHERE n.nspname = 'mahabodi' AND c.relkind = 'r'""").fetchone()[0])
    graph = float(c.execute("""SELECT coalesce(sum(pg_total_relation_size(c.oid)), 0) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                         WHERE n.nspname = %s AND c.relkind = 'r'""", ("mb_" + a.ns,)).fetchone()[0])
    res["bytes_relational_incl_vectors"] = int(rel * share)
    res["bytes_age_graph"] = int(graph)
    res["bytes_per_atf"] = round((rel * share + graph) / len(ids))
    res["text_bytes_per_atf"] = round(sum(len(snap["texts"].get(i, "").encode()) for i in ids) / len(ids))


if __name__ == "__main__":
    main()
