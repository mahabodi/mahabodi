"""PREREG_SCALE_V2 step 3 (and the full-scale part of step 2): entity linking with the shipped PostgreSQL store.

Phases (each writes its own result file; each resumable):
  bridge      test 100K pool, v1 test mentions: store M (exact dense, then IVFFlat at the dev-chosen probes) vs in-process M
  load_full   all 5,903,530 pages into namespace `full` (resumable by batch), build, ONE density pass (24 h budget),
              IVFFlat index (lists = ceil(sqrt(rows)), maintenance_work_mem 10GB)
  probes      full-scale probes on the 500 DEV mentions vs an exact NumPy scan of all stored passage vectors
  primary     fresh 1,000 (el_fresh_v2_ids.json) and the v1 test 1,000: store M vs L1 (Laya + dense top-20, n = 16)
  v1check     the v1 corner (mahabodi_pg.search on el_full, unchanged) on the v1 test items must equal bench_el_pg.json
  ablation    fresh items: {per-passage, first-passage (v1) vectors} x {MahaBodi cascade, PG text ranking}, RRF k = 60

Shortlist rule everywhere: k = 60 hits -> pages (pg_<row>_...) in rank order -> first 20; decider `decide`, n = 48.

    KILT_DIR=... python research/bench_el_store_v2.py --phase bridge --dsn "..."
"""
import argparse, glob, json, math, os, re, sys, time, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from store_parity_gate import wilson, mcnemar  # noqa: E402  (verbatim copies of bench.py's; registers the light 'bench')
from bench_el import Pages, laya_choice, ntitle, EL, R, ROOT, K  # noqa: E402

V1_DSN_DB = "el_full"
PAGE = re.compile(r"(?:F_)?pg_(\d+)_")


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


def run_store_arm(b, ms, P, ckp, mode="hybrid"):
    """store_query(k = 60) -> 20 pages -> decide n = 48, per mention, checkpointed."""
    ck = json.load(open(ckp)) if os.path.exists(ckp) else {}
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
    lat = [r["total_ms"] for r in rows]
    return score([r["pred"] for r in rows], [g in r["shortlist"] for r, g in zip(rows, gold)], gold,
                 {"latency_ms_p50": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
                  "retrieval_ms_p50": float(np.median([r["retrieval_ms"] for r in rows])), "shortlists": [r["shortlist"] for r in rows]})


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["bridge", "load_full", "probes", "primary", "v1check", "ablation"])
    ap.add_argument("--dsn", required=True, help="the store database (namespaces t100k, full)")
    ap.add_argument("--v1-dsn", default="host=127.0.0.1 port=5433 user=postgres dbname=el_full")
    ap.add_argument("--laya", default=os.path.join(ROOT, "models", "laya-v2"))
    ap.add_argument("--vector-type", default=None, help="halfvec or vector (default: from store_vector_index_dev100k.json)")
    a = ap.parse_args()
    from mahabodi import Bodi
    step2 = json.load(open(os.path.join(R, "store_vector_index_dev100k.json")))
    vtype = a.vector_type or ("halfvec" if step2["halfvec_use_at_5_9M"] else "vector")
    P = Pages()
    b = Bodi(); b.load_laya(a.laya, intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    out = {"phase": a.phase, "provenance": provenance(), "prereg": "research/PREREG_SCALE_V2.md", "vector_type": vtype, "t0": time.time()}
    fname = os.path.join(R, "el_store_v2_%s.json" % a.phase)

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
        t = time.time(); b.call("store_build_ivfflat_index", lists=math.ceil(math.sqrt(rows)), workers=6, maintenance_mem="10GB", probes=probes)
        times["ivfflat_build_s"] = round(time.time() - t, 1)
        import psycopg
        def idx_scans():  # the store's dense queries must use the IVFFlat index (store/pg.rs dense_tx), not an exact scan
            with psycopg.connect(a.dsn, autocommit=True) as sc:
                sc.execute("SELECT pg_stat_force_next_flush()")
                return sc.execute("SELECT coalesce(sum(idx_scan), 0) FROM pg_stat_user_indexes WHERE indexrelname = 'vec_ivfflat'").fetchone()[0]
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
        t = time.time(); b.call("store_build_ivfflat_index", lists=math.ceil(math.sqrt(rows)), workers=6, maintenance_mem="10GB", probes=10)
        times["ivfflat_build_s"] = round(time.time() - t, 1)
        out.update({"pages": n_pages, "passages": rows, "times": times, "stats": b.call("store_stats")})

    elif a.phase == "probes":
        import psycopg
        ms = load_mentions("dev")
        qs = np.asarray(b.embed_text([m["mention"] or m["state"] for m in ms]), dtype=np.float32)
        c = psycopg.connect(a.dsn, autocommit=True)
        # exact top-50 by streaming all stored passage vectors (NumPy), in chunks
        best_s = np.full((len(qs), 50), -1e9, dtype=np.float32); best_i = np.full((len(qs), 50), "", dtype=object)
        with c.cursor(name="v") as cur:
            cur.itersize = 200_000
            cur.execute("SELECT node_id, embedding::text FROM mahabodi_store.vec WHERE ns = 'full'")
            while True:
                chunk = cur.fetchmany(200_000)
                if not chunk:
                    break
                ids = np.array([r[0] for r in chunk], dtype=object)
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
            plan = "\n".join(r[0] for r in c.execute("EXPLAIN SELECT node_id FROM mahabodi_store.vec WHERE ns = 'full' ORDER BY embedding <#> %%s::text::%s LIMIT 50" % vtype,
                                                      ("[" + ",".join("%.7f" % x for x in qs[0]) + "]",)).fetchall())
            assert "Index Scan using vec_ivfflat" in plan, plan
            rec, lat = [], []
            for q, ex in zip(qs, exact):
                t = time.perf_counter()
                got = [r[0] for r in c.execute("SELECT node_id FROM mahabodi_store.vec WHERE ns = 'full' ORDER BY embedding <#> %%s::text::%s LIMIT 50" % vtype,
                                               ("[" + ",".join("%.7f" % x for x in q) + "]",)).fetchall()]
                lat.append((time.perf_counter() - t) * 1000); rec.append(len(set(got) & ex) / 50)
            sweep[str(p)] = {"recall_at_50": round(float(np.mean(rec)), 4), "latency_ms_p50": round(float(np.median(lat)), 1),
                             "latency_ms_p95": round(float(np.percentile(lat, 95)), 1), "plan": plan}
            print("probes", p, {k: x for k, x in sweep[str(p)].items() if k != "plan"}, flush=True)
        ok = [int(p) for p, r in sweep.items() if r["recall_at_50"] >= 0.98]
        out.update({"mention_ids": [m["id"] for m in ms], "sweep": sweep, "selected_probes": min(ok) if ok else 320, "reached_0_98": bool(ok)})

    elif a.phase == "primary":
        chk = extractor_self_check()
        assert chk["ok"], "v2 extractor does not reproduce mentions_mq.jsonl: %s" % chk
        out["extractor_self_check"] = chk
        probes = json.load(open(os.path.join(R, "el_store_v2_probes.json")))["selected_probes"]
        b.call("store_open", dsn=a.dsn, namespace="full", vector_type=vtype, create=False)
        b.call("store_set_probes", probes=probes)
        E = np.load(os.path.join(K, "dense", "emb.f16.npy"), mmap_mode="r")
        res = {}
        for label, ms in (("fresh", load_mentions("fresh")), ("v1_items", load_mentions("test"))):
            gold = [m["gold_row"] for m in ms]
            store = run_store_arm(b, ms, P, os.path.join(R, "el_store_v2_primary_%s.ckpt.json" % label))
            # L1: Laya on the dense top-20 over all pages (mention query), n = 16, as bench_el
            qv = np.asarray(b.embed_text([m["mention"] or m["state"] for m in ms]), dtype=np.float32)
            best_s = np.full((len(ms), 20), -1e4, dtype=np.float32); best_i = np.zeros((len(ms), 20), dtype=np.int64)
            for s0 in range(0, E.shape[0], 250_000):
                sc = qv @ np.asarray(E[s0:s0 + 250_000], dtype=np.float32).T
                top = np.argpartition(-sc, 20, axis=1)[:, :20]
                cs = np.concatenate([best_s, np.take_along_axis(sc, top, 1)], 1); ci = np.concatenate([best_i, top + s0], 1)
                j = np.argsort(-cs, axis=1)[:, :20]
                best_s, best_i = np.take_along_axis(cs, j, 1), np.take_along_axis(ci, j, 1)
            l1p = [laya_choice(b, m["state"], row.tolist(), P, 16, decide=False) for m, row in zip(ms, best_i)]
            l1 = score(l1p, [g in row.tolist() for g, row in zip(gold, best_i)], gold)
            res[label] = {"mention_ids": [m["id"] for m in ms], "gold_rows": gold, "store_M": store, "L1": l1,
                          "mcnemar_store_vs_L1": mcnemar([p == g for p, g in zip(store["pred"], gold)], [p == g for p, g in zip(l1p, gold)])}
            print(label, "store", store["accuracy"], "L1", l1["accuracy"], res[label]["mcnemar_store_vs_L1"], flush=True)
        out.update({"probes": probes, "results": res})

    elif a.phase in ("v1check", "ablation"):
        sys.path.insert(0, os.path.join(ROOT, "deploy", "sync"))
        import mahabodi_pg
        v1 = mahabodi_pg.connect(a.v1_dsn); v1.execute("SET hnsw.ef_search = 100")
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
            probes = json.load(open(os.path.join(R, "el_store_v2_probes.json")))["selected_probes"]
            b.call("store_open", dsn=a.dsn, namespace="full", vector_type=vtype, create=False)
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
                        "note": "descriptive; first_pgtext is the v1 retriever with harness RRF (v1check shows mahabodi_pg.search reproduces v1)"})
    out["seconds_total"] = round(time.time() - out.pop("t0"), 1)
    json.dump(out, open(fname, "w"), indent=1, default=str)
    print("PHASE-DONE", a.phase, flush=True)


if __name__ == "__main__":
    main()
