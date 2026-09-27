"""Memory compression vs retrieval on the same corpus: FastMemory (F), MahaBodi (M), MiniLM float32 flat (V32), TurboVec 4-bit
(TV4) and 2-bit (TV2). Pre-registered in research/PREREG_TURBOVEC.md (committed before any run).

Each (arm, size) runs in its own process (`--arm A --n N`), so peak RSS is per arm. With no --arm, the parent builds the
queries once, runs every arm as a child, and writes research/results/bench_turbovec.json with per-query ranks per arm.

    CUDA_VISIBLE_DEVICES= .venv/bin/python research/bench_turbovec.py --sizes 10000,30000,100000
"""
import argparse, hashlib, json, math, os, random, re, resource, subprocess, sys, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PAGES = "/media/sda/data/kilt/pages/part-0000.parquet"
WORK = "/media/sda/data/turbovec_bench"
ABSTRACT_CHARS = 2000  # kilt_prep.py
STOP = set("the a an and or of to in on at for by with from as is was were be been are that this which who whom whose it its "
           "his her their they he she we you i not no but also than then into over under about after before between during "
           "while when where there here such these those other more most some any all each both one two three first second".split())


def doc_text(title, abstract):
    return "# %s\n\n%s" % (title, abstract)


def load(n):
    return pq.read_table(PAGES, columns=["wikipedia_id", "title", "abstract", "paragraphs"]).slice(0, n).to_pydict()


def held_out_paragraph(paras):
    """The first paragraph NOT used in the abstract (same rule as kilt_prep.page_row)."""
    n = 0
    for i, p in enumerate(paras):
        if n >= ABSTRACT_CHARS:
            return p
        n += len(p) + 1
    return None


def words(s):
    return re.findall(r"[a-z][a-z0-9]{2,}", s.lower())


def build_queries(T, n, cap=1000):
    df = {}
    for ab, ti in zip(T["abstract"][:n], T["title"][:n]):
        for w in set(words(ti + " " + ab)):
            df[w] = df.get(w, 0) + 1
    cand = [i for i in range(n) if held_out_paragraph(T["paragraphs"][i])]
    rng = random.Random(0); rng.shuffle(cand)
    Q = []
    for i in cand[:cap]:
        hp = held_out_paragraph(T["paragraphs"][i])
        wid = T["wikipedia_id"][i]
        Q.append({"page": i, "wikipedia_id": wid, "type": "title", "q": T["title"][i]})
        sent = re.split(r"(?<=[.!?])\s+", hp.strip())[0]
        if 8 <= len(sent.split()) <= 40:
            Q.append({"page": i, "wikipedia_id": wid, "type": "sentence", "q": sent})
        kws = sorted({w for w in words(hp) if w not in STOP and df.get(w, 0) > 0}, key=lambda w: (df[w], w))[:3]
        if len(kws) >= 2:
            Q.append({"page": i, "wikipedia_id": wid, "type": "keywords", "q": " ".join(kws)})
    return Q


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


# ---------------- arms (each runs in its own process) ----------------

def arm_vectors(T, n, Q, bits):
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(8)
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"); st.max_seq_length = 256
    texts = [doc_text(t, a) for t, a in zip(T["title"][:n], T["abstract"][:n])]
    os.makedirs(WORK, exist_ok=True)
    ep = os.path.join(WORK, "minilm_%d.f32.npy" % n)
    hp = ep + ".texthash"
    th = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    t0 = time.time()
    if os.path.exists(ep) and os.path.exists(hp) and open(hp).read() == th:
        E = np.load(ep); embed_s = None  # reused (same texts, checked by hash); embedding time from the run that made it
    else:
        E = st.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(ep, E); open(hp, "w").write(th); embed_s = round(time.time() - t0, 1)
    qv = st.encode([x["q"] for x in Q], batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    t1 = time.time()
    if bits == 32:
        index_path = os.path.join(WORK, "v32_%d.npy" % n); np.save(index_path, E)
        search = lambda v: np.argsort(-(E @ v))[:10]
    else:
        from turbovec import TurboQuantIndex
        ix = TurboQuantIndex(dim=384, bit_width=bits); ix.add(E)
        index_path = os.path.join(WORK, "tv%d_%d.tv" % (bits, n)); ix.write(index_path)
        del E
        ix = TurboQuantIndex.load(index_path)
        search = lambda v: ix.search(v.reshape(1, -1), k=10)[1].reshape(-1)
    build_s = round(time.time() - t1, 1)
    ranks, lat = [], []
    for v, x in zip(qv, Q):
        t = time.perf_counter(); top = [int(i) for i in search(v)]; lat.append((time.perf_counter() - t) * 1000)
        ranks.append(top.index(x["page"]) + 1 if x["page"] in top else None)
    text_bytes = sum(len(s.encode()) for s in texts) + sum(len(s.encode()) for s in T["wikipedia_id"][:n])
    return {"stored_bytes_without_text": os.path.getsize(index_path), "text_and_id_bytes": text_bytes,
            "stored_bytes_with_text": os.path.getsize(index_path) + text_bytes, "embed_s": embed_s, "build_s": build_s,
            "ranks": ranks, "latency_ms_p50": pct(lat, .5), "latency_ms_p95": pct(lat, .95),
            "recoverable_text_without_text": 0.0, "recoverable_text_with_text": 1.0}


def arm_mahabodi(T, n, Q):
    from mahabodi import Bodi
    b = Bodi(); t0 = time.time()
    b.ingest_batch([{"text": doc_text(t, a), "source": "pg%d" % i} for i, (t, a) in enumerate(zip(T["title"][:n], T["abstract"][:n]))])
    build_s = round(time.time() - t0, 1)
    sd = b.snapshot()
    snap = json.dumps(sd, ensure_ascii=False).encode()
    snap_wo = json.dumps({k: v for k, v in sd.items() if k != "texts"}, ensure_ascii=False).encode()
    sp = os.path.join(WORK, "mahabodi_%d.snapshot.json" % n); os.makedirs(WORK, exist_ok=True); open(sp, "wb").write(snap)
    texts = sd["texts"]
    by_page = {}
    for aid, tx in texts.items():
        m = re.match(r"pg_(\d+)_", aid)
        if m:
            by_page.setdefault(int(m.group(1)), []).append(tx)
    norm = lambda s: " ".join(s.split())
    sample = sorted({x["page"] for x in Q})
    rec = sum(1 for p in sample if norm(T["abstract"][p])[:200] and norm(T["abstract"][p])[:200].split(" ")[0] and
              all(norm(s) in norm(" ".join(by_page.get(p, []))) for s in re.split(r"(?<=[.!?])\s+", norm(T["abstract"][p])) if s))
    ranks, lat = [], []
    for x in Q:
        t = time.perf_counter(); r = b.query(x["q"], k=10); lat.append((time.perf_counter() - t) * 1000)
        pages = []
        for h in r.get("hits", []):
            m = re.match(r"F_pg_(\d+)_", h["id"])
            if m and int(m.group(1)) not in pages:
                pages.append(int(m.group(1)))
        ranks.append(pages.index(x["page"]) + 1 if x["page"] in pages else None)
    return {"stored_bytes": len(snap), "stored_bytes_without_text": len(snap_wo), "stored_form": "snapshot JSON (atfs, links, texts, concepts); graph+index rebuilt on restore",
            "build_s": build_s, "ranks": ranks, "latency_ms_p50": pct(lat, .5), "latency_ms_p95": pct(lat, .95),
            "recoverable_text": rec / len(sample), "recoverable_id": 1.0, "recoverable_check": "every sentence of the abstract found in the page's snapshot texts"}


def arm_fastmemory(T, n, Q):
    import fastmemory
    from probe_fastmemory import atf_md
    md = atf_md(T["wikipedia_id"][:n], T["title"][:n], T["abstract"][:n])
    t0 = time.time(); s = fastmemory.process_markdown(md); build_s = round(time.time() - t0, 1)
    s = s if isinstance(s, str) else json.dumps(s)
    sp = os.path.join(WORK, "fastmemory_%d.json" % n); os.makedirs(WORK, exist_ok=True); open(sp, "w").write(s)
    sample = sorted({x["page"] for x in Q})
    rec_t = sum(1 for p in sample if T["abstract"][p].replace("\n", " ")[:200] in s) / len(sample)
    rec_i = sum(1 for p in sample if ("ATF_%s" % T["wikipedia_id"][p]) in s or ("Process_%s" % re.sub(r"\W+", "_", T["title"][p])) in s) / len(sample)
    # fastmemory's own search (mahabodi_core::query::fastmemory_search, ported): latency only, no page identity to score
    blocks = json.loads(s)
    def has(v, q):
        if not isinstance(v, dict):
            return False
        if any(isinstance(v.get(k), str) and q in v[k].lower() for k in ("name", "action", "id")):
            return True
        return any(isinstance(v.get(k), list) and any(has(c, q) for c in v[k]) for k in ("nodes", "sub_blocks"))
    lat, nonempty = [], 0
    for x in Q:
        q = x["q"].lower(); t = time.perf_counter(); res = [b_ for b_ in blocks if has(b_, q)]; lat.append((time.perf_counter() - t) * 1000)
        nonempty += bool(res)
    return {"stored_bytes": len(s.encode()), "build_s": build_s, "ranks": None,
            "recall_note": "SECONDARY row (Python module, a different code path from the CLI): recall@k = 0.00 by construction, the output keeps no page identity (ids and titles dropped; verbs re-extracted as Function nodes)", "recall_at_k": 0.0,
            "queries_with_any_result": nonempty, "latency_ms_p50": pct(lat, .5), "latency_ms_p95": pct(lat, .95),
            "recoverable_text": rec_t, "recoverable_id": rec_i}


def child(arm, n, qpath, out):
    T = load(n); Q = json.load(open(qpath))
    fn = {"V32": lambda: arm_vectors(T, n, Q, 32), "TV4": lambda: arm_vectors(T, n, Q, 4), "TV2": lambda: arm_vectors(T, n, Q, 2),
          "M": lambda: arm_mahabodi(T, n, Q), "Fpy": lambda: arm_fastmemory(T, n, Q)}[arm]
    r = fn(); r["peak_rss_mb"] = round(rss_mb())
    json.dump(r, open(out, "w"))


def wilson(k, n):
    if n == 0:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000,30000,100000")
    ap.add_argument("--arms", default="M,V32,TV4,TV2,Fpy")
    ap.add_argument("--arm"); ap.add_argument("--n", type=int); ap.add_argument("--queries"); ap.add_argument("--child-out")
    ap.add_argument("--out", default=os.path.join(R, "bench_turbovec.json"))
    ap.add_argument("--export-fm", action="store_true", help="write the F-arm inputs (ATF markdown + queries) for the macOS CLI run")
    a = ap.parse_args()
    if a.arm:
        return child(a.arm, a.n, a.queries, a.child_out)
    if a.export_fm:
        from probe_fastmemory import atf_md
        d = os.path.join(R, "bench_turbovec_inputs"); os.makedirs(d, exist_ok=True)
        for n in [int(x) for x in a.sizes.split(",")]:
            T = load(n)
            open(os.path.join(d, "fm_atf_%d.md" % n), "w").write(atf_md(T["wikipedia_id"], T["title"], T["abstract"]))
            json.dump(build_queries(T, n), open(os.path.join(d, "queries_%d.json" % n), "w"))
            print("exported", n)
        return
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"sizes": {}}
    res.update({"protocol": "research/PREREG_TURBOVEC.md", "provenance": provenance()})
    os.makedirs(WORK, exist_ok=True)
    for n in [int(x) for x in a.sizes.split(",")]:
        T = load(n)
        raw = sum(len((t + ab).encode()) for t, ab in zip(T["title"], T["abstract"]))
        qp = os.path.join(WORK, "queries_%d.json" % n)
        Q = build_queries(T, n); json.dump(Q, open(qp, "w"))
        S = res["sizes"].setdefault(str(n), {"raw_source_bytes": raw, "queries": Q, "arms": {}})
        for arm in a.arms.split(","):
            if arm in S["arms"]:
                continue
            co = os.path.join(WORK, "arm_%s_%d.json" % (arm, n))
            t0 = time.time()
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--arm", arm, "--n", str(n), "--queries", qp, "--child-out", co],
                               capture_output=True, text=True)
            if p.returncode != 0 or not os.path.exists(co):
                S["arms"][arm] = {"failed": True, "returncode": p.returncode, "stderr_tail": p.stderr[-2000:]}
            else:
                r = json.load(open(co)); r["wall_s"] = round(time.time() - t0, 1)
                if r.get("ranks") is not None:
                    for typ in ("title", "sentence", "keywords"):
                        idx = [i for i, x in enumerate(Q) if x["type"] == typ]
                        for k in (1, 5, 10):
                            hit = sum(1 for i in idx if r["ranks"][i] is not None and r["ranks"][i] <= k)
                            r.setdefault("recall", {}).setdefault(typ, {})["@%d" % k] = {"hits": hit, "n": len(idx),
                                                                                      "recall": round(hit / max(1, len(idx)), 4), "ci95": wilson(hit, len(idx))}
                for key in ("stored_bytes", "stored_bytes_without_text", "stored_bytes_with_text"):
                    if key in r:
                        r["ratio_raw_over_" + key] = round(raw / r[key], 3)
                S["arms"][arm] = r
            print(n, arm, {k: v for k, v in S["arms"][arm].items() if k not in ("ranks", "recall", "stderr_tail")}, flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
