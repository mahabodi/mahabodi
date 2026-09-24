"""MahaBodi <-> PostgreSQL (Apache AGE + pgvector) sync, search and hydration.

Reference implementation for the patterns in Enterprise.md. It moves memory between an
in-process MahaBodi engine (`Bodi.snapshot()` / `Bodi.restore()`) and the shared store defined
in deploy/postgres/01_schema.sql:

  load_snapshot()   MahaBodi snapshot (+ optional embeddings) -> relational rows + AGE graph
  search()          hybrid retrieval in SQL: lexical (tsvector) + typo-corrected lexical
                    (pg_trgm vocabulary) + dense (pgvector), fused by reciprocal rank
  neighbourhood()   Cypher: ATFs that share a Data/Access/Event/Concept node with the seeds
  hydrate()         selected ATFs -> a MahaBodi snapshot, ready for Bodi.restore()

It is exercised end to end by deploy/postgres/test_store.py. It is NOT part of the MahaBodi
engine: MahaBodi itself keeps memory in process; see Enterprise.md for what is and isn't built.

Requires: psycopg >= 3.2 (pip install "psycopg[binary]").
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Sequence

import psycopg
from psycopg.adapt import Dumper

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = set("a an and are as at be but by for from has have in is it its of on or that the this to was were will with".split())


class _Agtype(str):
    """A string bound as an AGE `agtype` parameter (AGE requires a real bind parameter)."""


def connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("LOAD 'age'")
    conn.execute("SET search_path = ag_catalog, mahabodi, public")
    oid = conn.execute("SELECT 'ag_catalog.agtype'::regtype::oid").fetchone()[0]

    class AgDumper(Dumper):
        def dump(self, obj):
            return obj.encode()
    AgDumper.oid = oid
    conn.adapters.register_dumper(_Agtype, AgDumper)
    return conn


def _graph(conn, ns: str) -> str:
    row = conn.execute("SELECT graph FROM mahabodi.namespace WHERE name = %s", (ns,)).fetchone()
    if row is None:
        raise KeyError("unknown namespace %r (call mahabodi.create_namespace first)" % ns)
    return row[0]


def _cypher(conn, graph: str, query: str, params: dict | None = None, cols: str = "a agtype"):
    # graph comes from the namespace table (validated name), never from user input
    sql = "SELECT * FROM cypher('%s', $$ %s $$%s) AS (%s)" % (graph, query, ", %s" if params is not None else "", cols)
    return conn.execute(sql, (_Agtype(json.dumps(params)),) if params is not None else None).fetchall()


def _terms(text: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD.findall(text)) if len(w) > 2 and w not in _STOP}


def _batches(xs: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def load_snapshot(conn, ns: str, snap: dict, embeddings: dict[str, list[float]] | None = None,
                  model: str = "", source: str | None = None, batch: int = 500) -> dict:
    """Upsert a MahaBodi snapshot into namespace `ns`. Idempotent per ATF id.

    For bulk loads (millions of ATFs) prefer COPY into the relational tables and AGE's
    load_labels_from_file / load_edges_from_file; this function uses batched UNWIND/MERGE,
    which is simple and correct but not the fastest path."""
    graph = _graph(conn, ns)
    atfs, texts = snap.get("atfs", []), snap.get("texts", {})
    vocab: dict[str, int] = {}
    with conn.transaction():
        for chunk in _batches(atfs, batch):
            rows = []
            for a in chunk:
                body = texts.get(a["id"], "")
                rows.append((ns, a["id"], a.get("action", ""), a.get("input", ""), a.get("logic", ""),
                             a.get("access", ""), a.get("events", ""), list(a.get("data_connections", [])),
                             body, source, hashlib.sha256(body.encode()).digest()))
                for t in _terms(" ".join([a["id"], a.get("action", ""), body])):
                    vocab[t] = vocab.get(t, 0) + 1
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO mahabodi.atf (namespace, id, action, input, logic, access, events, data_connections, body, source, content_hash) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (namespace, id) DO UPDATE SET "
                    "action=EXCLUDED.action, input=EXCLUDED.input, logic=EXCLUDED.logic, access=EXCLUDED.access, events=EXCLUDED.events, "
                    "data_connections=EXCLUDED.data_connections, body=EXCLUDED.body, source=EXCLUDED.source, "
                    "content_hash=EXCLUDED.content_hash, updated_at=now()", rows)
            # graph: Function vertices, shared Data/Access/Event vertices, typed edges
            fx = [{"id": a["id"], "action": a.get("action", ""),
                   "data": list(a.get("data_connections", [])),
                   "access": [x.strip() for x in a.get("access", "").split(",") if x.strip()],
                   "events": [x.strip() for x in a.get("events", "").split(",") if x.strip()]} for a in chunk]
            _cypher(conn, graph, "UNWIND $rows AS r MERGE (f:Function {id: r.id}) SET f.action = r.action", {"rows": fx})
            for label, key, rel in (("Data", "data", "USES"), ("Access", "access", "GRANTS"), ("Event", "events", "EMITS")):
                pairs = [{"f": r["id"], "n": n} for r in fx for n in r[key]]
                if pairs:
                    _cypher(conn, graph, "UNWIND $rows AS r MATCH (f:Function {id: r.f}) MERGE (x:%s {name: r.n}) "
                                         "MERGE (f)-[:%s]->(x)" % (label, rel), {"rows": pairs})
        links = [{"a": a, "b": b} for a, b in snap.get("links", [])]
        if links:
            _cypher(conn, graph, "UNWIND $rows AS r MATCH (a:Function {id: r.a}), (b:Function {id: r.b}) MERGE (a)-[:LINKS]->(b)", {"rows": links})
        concepts = [{"f": f, "k": k} for f, k in snap.get("concepts", [])]
        if concepts:
            _cypher(conn, graph, "UNWIND $rows AS r MATCH (f:Function {id: r.f}) MERGE (k:Concept {name: r.k}) MERGE (f)-[:MENTIONS]->(k)", {"rows": concepts})
        if vocab:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO mahabodi.vocab (namespace, term, doc_freq) VALUES (%s,%s,%s) "
                                "ON CONFLICT (namespace, term) DO UPDATE SET doc_freq = mahabodi.vocab.doc_freq + EXCLUDED.doc_freq",
                                [(ns, t, n) for t, n in vocab.items()])
        if embeddings:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO mahabodi.atf_embedding (namespace, atf_id, model, embedding) VALUES (%s,%s,%s,%s::vector) "
                                "ON CONFLICT (namespace, atf_id, model) DO UPDATE SET embedding = EXCLUDED.embedding",
                                [(ns, i, model, "[" + ",".join("%.7g" % x for x in v) + "]") for i, v in embeddings.items()])
    refresh_degrees(conn, ns)
    return {"atfs": len(atfs), "links": len(links), "concepts": len(concepts), "vocab_terms": len(vocab),
            "embeddings": len(embeddings or {})}


def refresh_degrees(conn, ns: str) -> None:
    """Store each shared vertex's degree (how many ATFs use it) as a `df` property.

    neighbourhood() skips vertices above a degree cap. In prose memories the data connections are
    content terms, and a common term links thousands of ATFs: expanding through it makes a 2-hop
    query touch most of the graph (measured: one query on 4,897 ATFs ran > 20 minutes without the cap)."""
    graph = _graph(conn, ns)
    for label, rel in (("Data", "USES"), ("Access", "GRANTS"), ("Event", "EMITS"), ("Concept", "MENTIONS")):
        _cypher(conn, graph, "MATCH (f:Function)-[:%s]->(x:%s) WITH x, count(f) AS n SET x.df = n" % (rel, label))


def correct_terms(conn, ns: str, query: str, min_sim: float = 0.4) -> list[str]:
    """Typo correction: each query term not in the vocabulary -> its most similar known term."""
    out = []
    for t in sorted(_terms(query)):
        row = conn.execute("SELECT term, similarity(term, %s) AS s FROM mahabodi.vocab WHERE namespace = %s AND term %% %s "
                           "ORDER BY s DESC, doc_freq DESC LIMIT 1", (t, ns, t)).fetchone()
        if row and row[0] != t and row[1] >= min_sim:
            out.append(row[0])
    return out


def search(conn, ns: str, query: str, qvec: list[float] | None = None, k: int = 5, pool: int = 50,
           rrf_k: int = 60, model: str = "") -> list[dict]:
    """Hybrid retrieval, fused by reciprocal rank (sum over stages of 1 / (rrf_k + rank))."""
    corrected = " ".join(correct_terms(conn, ns, query))
    lexical_q = query + (" " + corrected if corrected else "")
    vec = "[" + ",".join("%.7g" % x for x in qvec) + "]" if qvec is not None else None
    rows = conn.execute(
        """
        WITH lex AS (
            SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC, id) AS r
            FROM mahabodi.atf, websearch_to_tsquery('english', %(lq)s) q0,
                 LATERAL (SELECT to_tsquery('english', replace(q0::text, '&', '|')) AS q) qq
            WHERE namespace = %(ns)s AND tsv @@ q
            ORDER BY r LIMIT %(pool)s),
        dense AS (
            SELECT atf_id AS id, row_number() OVER (ORDER BY embedding <=> %(vec)s::vector, atf_id) AS r,
                   1 - (embedding <=> %(vec)s::vector) AS sim
            FROM mahabodi.atf_embedding
            WHERE %(vec)s::text IS NOT NULL AND namespace = %(ns)s AND model = %(model)s
            ORDER BY embedding <=> %(vec)s::vector LIMIT %(pool)s),
        fused AS (
            SELECT id, sum(1.0 / (%(rrf)s + r)) AS score,
                   bool_or(src = 'lex') AS lexical, max(sim) AS dense_sim
            FROM (SELECT id, r, 'lex' AS src, NULL::float8 AS sim FROM lex
                  UNION ALL SELECT id, r, 'dense', sim FROM dense) u
            GROUP BY id)
        SELECT f.id, f.score, f.lexical, f.dense_sim, a.action, left(a.body, 200)
        FROM fused f JOIN mahabodi.atf a ON a.namespace = %(ns)s AND a.id = f.id
        ORDER BY f.score DESC, f.id LIMIT %(k)s
        """,
        {"lq": lexical_q, "ns": ns, "pool": pool, "vec": vec, "model": model, "rrf": rrf_k, "k": k}).fetchall()
    return [{"id": r[0], "score": float(r[1]), "lexical": r[2], "dense_similarity": r[3], "action": r[4],
             "snippet": r[5], "corrected_terms": corrected.split() if corrected else []} for r in rows]


def neighbourhood(conn, ns: str, atf_ids: list[str], limit: int = 200, max_degree: int = 25) -> list[str]:
    """ATF ids sharing a specific (low-degree) Data/Access/Event/Concept node, or a LINKS edge,
    with any seed ATF. Shared vertices used by more than `max_degree` ATFs are generic hubs and
    are skipped (see refresh_degrees; vertices without a `df` property are skipped too)."""
    graph = _graph(conn, ns)
    rows = _cypher(conn, graph,
                   "UNWIND $ids AS s MATCH (a:Function {id: s})-[]-(x) WHERE x.df <= %d "
                   "MATCH (x)-[]-(b:Function) WHERE b.id <> s RETURN DISTINCT b.id LIMIT %d" % (int(max_degree), int(limit)),
                   {"ids": atf_ids}, cols="id agtype")
    rows += _cypher(conn, graph, "UNWIND $ids AS s MATCH (a:Function {id: s})-[:LINKS]-(b:Function) RETURN DISTINCT b.id",
                    {"ids": atf_ids}, cols="id agtype")
    return sorted({json.loads(r[0]) for r in rows} - set(atf_ids))


def hydrate(conn, ns: str, atf_ids: list[str]) -> dict:
    """Build a MahaBodi snapshot (Bodi.restore format, version 1) holding exactly these ATFs."""
    graph = _graph(conn, ns)
    rows = conn.execute("SELECT id, action, input, logic, data_connections, access, events, body FROM mahabodi.atf "
                        "WHERE namespace = %s AND id = ANY(%s) ORDER BY id", (ns, list(atf_ids))).fetchall()
    ids = [r[0] for r in rows]
    links = [(json.loads(a), json.loads(b)) for a, b in _cypher(
        conn, graph, "UNWIND $ids AS s MATCH (a:Function {id: s})-[:LINKS]->(b:Function) RETURN a.id, b.id",
        {"ids": ids}, cols="a agtype, b agtype") if json.loads(b) in set(ids)]
    concepts = [(json.loads(f), json.loads(c)) for f, c in _cypher(
        conn, graph, "UNWIND $ids AS s MATCH (f:Function {id: s})-[:MENTIONS]->(k:Concept) RETURN f.id, k.name",
        {"ids": ids}, cols="f agtype, c agtype")]
    return {"version": 1,
            "atfs": [{"id": r[0], "action": r[1], "input": r[2], "logic": r[3], "data_connections": list(r[4]),
                      "access": r[5], "events": r[6]} for r in rows],
            "links": [list(x) for x in links], "texts": {r[0]: r[7] for r in rows}, "concepts": [list(x) for x in concepts]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--ns", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("load"); p.add_argument("snapshot_json")
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=5)
    p = sub.add_parser("hydrate"); p.add_argument("ids", nargs="+")
    a = ap.parse_args()
    c = connect(a.dsn)
    if a.cmd == "load":
        print(json.dumps(load_snapshot(c, a.ns, json.load(open(a.snapshot_json)))))
    elif a.cmd == "search":
        print(json.dumps(search(c, a.ns, a.query, k=a.k), indent=1))
    else:
        print(json.dumps(hydrate(c, a.ns, a.ids)))
