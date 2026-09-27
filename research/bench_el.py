"""Entity linking at 10K / 100K / 5.9M candidates (research/PREREG_MILLION_SCALE.md + Addendum 1 + pre-run clarification 1).

Arms (every arm sees the same shared pool per stage):
  L0   Laya alone with all N options in one call: cannot run at N >= 10K (reported with the reason, never as 0 %)
  D    dense MiniLM top-1 within the pool (the shared exact index)
  BM25 BM25 top-1 within the pool (the shared bm25s index)
  P0   context-free pool-bias control: popularity prior (the most frequent gold for the mention string in AIDA train,
       dev mentions excluded), else the exact normalised title match, else nothing
  K    kNN over AIDA-train mention states (dev mentions excluded): the nearest training mentions vote for their gold
  L1   MiniLM dense shortlist (top-k within the pool) -> Laya decides (Laya-identical `predict`, no tournament)
  M    MahaBodi: hybrid memory holding the pool pages -> `query` shortlist (top-k pages) -> `decide` (tournament)
  L1'  Laya (`predict`) on M's shortlist: separates MahaBodi's retrieval from its decision
Options: page title as the key, "title: first N tokens of the abstract" as the description (N fixed on dev, the same for
L1, L1' and M). k is tuned on dev per arm (L1, M) on the dev 100K pool. Test is scored once with those settings.
The full-stage M arm needs the PostgreSQL path (in-process memory cannot hold 5.9M pages); until then it is reported
"not run" with that reason.

    .venv/bin/python research/bench_el.py --phase tune   -> research/results/bench_el_tune.json
    .venv/bin/python research/bench_el.py --phase test   -> research/results/bench_el.json
"""
import argparse, glob, json, math, os, re, sys, time
from collections import Counter, defaultdict
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from bench import wilson, mcnemar  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
K = "/media/sda/data/kilt"; EL = os.path.join(K, "el")
QUESTION = "Which Wikipedia entity does the mention marked by [START_ENT] ... [END_ENT] in `text` refer to?"
K_GRID, N_GRID = [5, 10, 20, 50], [16, 48]


def ntitle(t):
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", t.lower())).strip()


class Pages:
    def __init__(self):
        self.shards = sorted(glob.glob(os.path.join(K, "pages", "part-*.parquet")))
        self.titles, self.bounds = [], []
        for s in self.shards:
            t = pq.read_table(s, columns=["title"]).column("title").to_pylist()
            self.bounds.append((len(self.titles), len(self.titles) + len(t))); self.titles.extend(t)
        self._abs = {}

    def abstracts(self, rows):
        need = sorted(set(int(r) for r in rows) - set(self._abs))
        by = defaultdict(list)
        for r in need:
            for si, (a, b) in enumerate(self.bounds):
                if a <= r < b:
                    by[si].append(r); break
        for si, rs in by.items():
            a = self.bounds[si][0]
            col = pq.read_table(self.shards[si], columns=["abstract"]).column("abstract")
            for r in rs:
                self._abs[r] = col[r - a].as_py()
        return {r: self._abs[int(r)] for r in rows}


def option_text(title, abstract, ntok):
    return "%s: %s" % (title, " ".join(abstract.split()[:ntok]))


def laya_choice(b, state, rows, P, ntok, decide):
    """Laya (predict) or MahaBodi (decide) choosing among candidate pages; -> chosen row or None."""
    rows = list(dict.fromkeys(int(r) for r in rows))
    if not rows:
        return None
    ab = P.abstracts(rows)
    keys, crit = [], {}
    for r in rows:
        k = P.titles[r]
        if k in crit:  # duplicate titles: keep the first
            continue
        keys.append((k, r)); crit[k] = option_text(P.titles[r], ab[r], ntok)
    q = {"entity": {"type": "choice", "instructions": QUESTION, "criteria": crit}}
    st = {"text": state}
    out = b.decide(st, q) if decide else b.predict(st, q)
    a = out["answers"]["entity"]
    probs = a.get("probabilities") or {}
    if not probs:
        return None
    best = max(probs, key=probs.get)
    return dict(keys).get(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tune", "test"], required=True)
    ap.add_argument("--stages", default="10000,100000,full")
    ap.add_argument("--laya", default=os.path.join(ROOT, "models", "laya-v2"))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N mentions only (results not for reporting)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--tag", default="_mq", help="mention-mined files (clarification 2)")
    a = ap.parse_args()
    from mahabodi import Bodi
    import torch
    from sentence_transformers import SentenceTransformer
    t_start = time.time()
    M = [json.loads(l) for l in open(os.path.join(EL, "mentions%s.jsonl" % a.tag))]
    split = "dev" if a.phase == "tune" else "test"
    ms = [m for m in M if m["split"] == split]
    if a.limit:
        ms = ms[:a.limit]
    dev_ids = {m["id"] for m in M if m["split"] == "dev"}
    P = Pages()
    E = np.load(os.path.join(K, "dense", "emb.f16.npy"), mmap_mode="r")
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"); st.max_seq_length = 256
    qtext = [m["mention"] or m["state"] for m in ms]  # clarification 2: retrieve by mention, decide with the context state
    Qv = st.encode(qtext, batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    Qs = st.encode([m["state"] for m in ms], batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)  # K: context kNN
    # AIDA train (dev mentions excluded) for P0 prior and K
    train = []
    for line in open(os.path.join(K, "aidayago2-train-kilt.jsonl")):
        d = json.loads(line)
        if d["id"] in dev_ids or not d["output"] or not d["output"][0].get("provenance"):
            continue
        from kilt_el_prep import state_of
        train.append((d["meta"].get("mention") or "", state_of(d["input"]), str(d["output"][0]["provenance"][0]["wikipedia_id"])))
    ids = np.load(os.path.join(K, "dense", "ids.npy")); row_of = {str(x): i for i, x in enumerate(ids)}
    prior = defaultdict(Counter)
    for men, _, gid in train:
        if gid in row_of:
            prior[men.lower().strip()][row_of[gid]] += 1
    title_rows = defaultdict(list)
    for i, t in enumerate(P.titles):
        title_rows[ntitle(t)].append(i)
    Tv = st.encode([s for _, s, _ in train], batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    Ty = [row_of.get(g) for _, _, g in train]
    lb = Bodi(); lb.load_laya(a.laya, intra_threads=8)  # Laya-identical predict

    def pool_rows(stage):
        return None if stage == "full" else np.load(os.path.join(EL, "pool_%s_%s%s.npy" % (split, stage, a.tag)))

    res = {"phase": a.phase, "split": split, "n": len(ms), "mention_ids": [m["id"] for m in ms], "gold_rows": [m["gold_row"] for m in ms], "question": QUESTION, "provenance": provenance(), "stages": {},
           "P0_source": "mention->entity counts from aidayago2-train-kilt.jsonl only (no testa/testb strings), with the 500 dev-sample mentions excluded; %d training mentions used" % len(train),
           "mentions_file": "mentions%s.jsonl" % a.tag,
           "selection_rule": "max dev accuracy on the dev 100K pool, ties to smaller k then smaller n; applied unchanged at every stage (PREREG_MILLION_SCALE.md clarification 3, commit b4ef6eb)"}
    tune = json.load(open(os.path.join(R, "bench_el_tune.json"))) if a.phase == "test" else None
    for stage in a.stages.split(","):
        if a.phase == "tune" and stage != "100000":
            continue
        pool = pool_rows(stage)
        inpool = (lambda r: True) if pool is None else (lambda r, s=set(pool.tolist()): r in s)
        S = {"N": 5903530 if pool is None else len(pool), "arms": {}}
        gold = [m["gold_row"] for m in ms]
        # dense ranking within the pool
        t0 = time.time()
        if pool is None:
            dense_rank = [m["dense_top"] for m in ms]
        else:
            Pe = np.asarray(E[pool], dtype=np.float32)
            sc = Qv @ Pe.T
            top = np.argsort(-sc, axis=1)[:, :max(K_GRID)]
            dense_rank = [pool[t].tolist() for t in top]
        dense_s = time.time() - t0
        # BM25 within the pool: filter the mined top list, else score the pool
        if pool is None:
            bm_rank = [m["bm25_top"] for m in ms]
        else:
            import bm25s, Stemmer
            r_ = getattr(main, "_bm", None) or bm25s.BM25.load(os.path.join(K, "bm25")); main._bm = r_
            mask = np.zeros(E.shape[0], dtype=np.float32); mask[pool] = 1.0
            tok = bm25s.tokenize(qtext, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)
            br, _ = r_.retrieve(tok, k=max(K_GRID), weight_mask=mask, show_progress=False, n_threads=8)
            bm_rank = [[int(x) for x in row] for row in br]
        pred = {"D": [r[0] if r else None for r in dense_rank], "BM25": [r[0] if r else None for r in bm_rank]}
        # P0
        p0 = []
        for m in ms:
            men = (m.get("mention") or "").lower().strip()
            c = [(n, r) for r, n in prior.get(men, Counter()).most_common() if inpool(r)]
            if c:
                p0.append(c[0][1]); continue
            tm = [r for r in title_rows.get(ntitle(men), []) if inpool(r)]
            p0.append(tm[0] if tm else None)
        pred["P0"] = p0
        # K: kNN over train states, vote restricted to the pool (k, T fixed: 10, 0.05 - tuned on dev below when phase=tune)
        simk = Qs @ Tv.T
        order = np.argsort(-simk, axis=1)[:, :50]
        def knn(kk, kt):
            kp = []
            for i in range(len(ms)):
                nn = order[i][:kk]; vote = Counter()
                for j in nn:
                    if Ty[j] is not None and inpool(Ty[j]):
                        vote[Ty[j]] += math.exp((simk[i, j] - simk[i, nn[0]]) / kt)
                kp.append(vote.most_common(1)[0][0] if vote else None)
            return kp
        if a.phase == "tune":
            S["tune_K"] = {"%d_%s" % (kk, kt): round(float(np.mean([p == g for p, g in zip(knn(kk, kt), [m["gold_row"] for m in ms])])), 4)
                           for kk in (1, 5, 10, 25) for kt in (0.02, 0.05, 0.1)}
        kk, kt = (tune["K"]["k"], tune["K"]["T"]) if tune else (10, 0.05)
        pred["K"] = knn(kk, kt)
        # L1 / M / L1'
        grid = [(k, n) for k in K_GRID for n in N_GRID] if a.phase == "tune" else None
        def run_l1(k, n):
            lat, out = [], []
            for m, dr in zip(ms, dense_rank):
                t = time.perf_counter(); out.append(laya_choice(lb, m["state"], dr[:k], P, n, decide=False)); lat.append((time.perf_counter() - t) * 1000)
            return out, lat
        ckp = os.path.join(R, "bench_el_tune.ckpt.json")  # resumable tuning: each (k, n) is saved as it finishes
        ck = json.load(open(ckp)) if (a.phase == "tune" and os.path.exists(ckp)) else {"L1": {}, "M": {}}
        def save_ck():
            if a.phase == "tune":
                json.dump(ck, open(ckp + ".tmp", "w")); os.replace(ckp + ".tmp", ckp)
        if a.phase == "tune":
            tl = {}
            tl_detail = {}
            for k, n in grid:
                key = "%d_%d" % (k, n)
                if key in ck["L1"]:
                    tl_detail[key] = ck["L1"][key]; tl[key] = tl_detail[key]["accuracy"]
                    print("tune L1", k, n, "(resumed from checkpoint)", flush=True); continue
                o, _ = run_l1(k, n); tl["%d_%d" % (k, n)] = round(float(np.mean([p == g for p, g in zip(o, gold)])), 4)
                ins = [g in dr[:k] for g, dr in zip(gold, dense_rank)]
                tl_detail["%d_%d" % (k, n)] = {"accuracy": tl["%d_%d" % (k, n)], "shortlist_recall": round(float(np.mean(ins)), 4),
                                               "accuracy_given_gold_in_shortlist": round(float(np.mean([p == g for p, g, i in zip(o, gold, ins) if i])), 4) if any(ins) else None,
                                               "ci95": wilson(sum(p == g for p, g in zip(o, gold)), len(gold)), "pred": o, "shortlist_hit": ins}
                ck["L1"][key] = tl_detail[key]; save_ck()
                print("tune L1", k, n, {x: y for x, y in tl_detail["%d_%d" % (k, n)].items() if x not in ("pred", "shortlist_hit")}, flush=True)
            S["tune_L1"] = tl; S["tune_L1_detail"] = tl_detail
        else:
            k1, n1 = tune["L1"]["k"], tune["L1"]["n"]
            o, lat = run_l1(k1, n1); pred["L1"] = o
            S["arms"]["L1_latency_ms_p50"] = float(np.median(lat)); S["L1_settings"] = {"k": k1, "n": n1}
            S["L1_shortlist_recall"] = round(float(np.mean([g in dr[:k1] for g, dr in zip(gold, dense_rank)])), 4)
        if pool is not None and a.phase == "tune" and len(ck["M"]) == len(K_GRID) * len(N_GRID):
            S["tune_M"] = {kk: v["accuracy"] for kk, v in ck["M"].items()}; S["tune_M_detail"] = ck["M"]
            print("tune M: all settings resumed from checkpoint", flush=True)
        elif pool is not None:
            b = Bodi(); b.load_laya(a.laya, intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
            t0 = time.time()
            ab = P.abstracts(pool.tolist())
            b.ingest_batch([{"text": "# %s\n\n%s" % (P.titles[r], ab[r]), "source": "pg%d" % r} for r in pool.tolist()])
            S["M_build_s"] = round(time.time() - t0, 1)
            def m_short(query, k):
                hits = b.query(query, k=3 * k).get("hits", [])
                rows = []
                for h in hits:
                    mm = re.match(r"F_pg_(\d+)_", h["id"])
                    if mm and int(mm.group(1)) not in rows:
                        rows.append(int(mm.group(1)))
                return rows[:k]
            def run_m(k, n, laya_only=False):
                lat, out, rec = [], [], []
                for m in ms:
                    t = time.perf_counter(); sh = m_short(m["mention"] or m["state"], k)
                    out.append(laya_choice(lb if laya_only else b, m["state"], sh, P, n, decide=not laya_only)); lat.append((time.perf_counter() - t) * 1000)
                    rec.append(m["gold_row"] in sh)
                return out, lat, rec
            if a.phase == "tune":
                tm = {}
                tm_detail = {}
                for k, n in grid:
                    key = "%d_%d" % (k, n)
                    if key in ck["M"]:
                        tm_detail[key] = ck["M"][key]; tm[key] = tm_detail[key]["accuracy"]
                        print("tune M", k, n, "(resumed from checkpoint)", flush=True); continue
                    o, _, rec = run_m(k, n); tm["%d_%d" % (k, n)] = round(float(np.mean([p == g for p, g in zip(o, gold)])), 4)
                    tm_detail["%d_%d" % (k, n)] = {"accuracy": tm["%d_%d" % (k, n)], "shortlist_recall": round(float(np.mean(rec)), 4),
                                                   "accuracy_given_gold_in_shortlist": round(float(np.mean([p == g for p, g, i in zip(o, gold, rec) if i])), 4) if any(rec) else None,
                                                   "ci95": wilson(sum(p == g for p, g in zip(o, gold)), len(gold)), "pred": o, "shortlist_hit": rec}
                    ck["M"][key] = tm_detail[key]; save_ck()
                    print("tune M", k, n, {x: y for x, y in tm_detail["%d_%d" % (k, n)].items() if x not in ("pred", "shortlist_hit")}, flush=True)
                S["tune_M"] = tm; S["tune_M_detail"] = tm_detail
            else:
                km, nm = tune["M"]["k"], tune["M"]["n"]
                o, lat, rec = run_m(km, nm); pred["M"] = o
                S["M_latency_ms_p50"] = float(np.median(lat)); S["M_settings"] = {"k": km, "n": nm}; S["M_shortlist_recall"] = round(float(np.mean(rec)), 4)
                o2, _, _ = run_m(km, nm, laya_only=True); pred["L1p"] = o2
            del b
        else:
            S["M_not_run"] = "full 5.9M stage: in-process memory cannot hold 5.9M pages (probe_memory_scale); needs the PostgreSQL path"
        S["L0"] = "cannot run: %d options exceed Laya's context (one shared token budget for all options)" % S["N"]
        for arm, p in pred.items():
            c = [x == g for x, g in zip(p, gold)]
            S["arms"][arm] = {"accuracy": round(float(np.mean(c)), 4), "ci95": wilson(sum(c), len(c)), "answered": sum(x is not None for x in p), "pred": p}
        if "L1" in pred:
            cl = [x == g for x, g in zip(pred["L1"], gold)]
            for arm in pred:
                if arm != "L1":
                    S["arms"][arm]["mcnemar_vs_L1"] = mcnemar([x == g for x, g in zip(pred[arm], gold)], cl)
            S["pool_biased"] = S["arms"]["P0"]["accuracy"] >= S["arms"]["L1"]["accuracy"]
        S["dense_rank_s"] = round(dense_s, 1)
        res["stages"][stage] = S
        print(stage, {k: v["accuracy"] for k, v in S["arms"].items() if isinstance(v, dict) and "accuracy" in v}, flush=True)
    if a.phase == "tune":
        S = res["stages"]["100000"]
        bk = lambda d: max(d, key=lambda x: (d[x], -int(x.split("_")[0]), -int(x.split("_")[1])))  # ties: smaller k, then smaller n
        l1, mm = bk(S["tune_L1"]), bk(S["tune_M"])
        res["L1"] = {"k": int(l1.split("_")[0]), "n": int(l1.split("_")[1]), "dev_accuracy": S["tune_L1"][l1]}
        res["M"] = {"k": int(mm.split("_")[0]), "n": int(mm.split("_")[1]), "dev_accuracy": S["tune_M"][mm]}
        kb = max(S["tune_K"], key=lambda x: (S["tune_K"][x], -int(x.split("_")[0])))
        res["K"] = {"k": int(kb.split("_")[0]), "T": float(kb.split("_")[1]), "dev_accuracy": S["tune_K"][kb]}
    res["seconds"] = round(time.time() - t_start, 1)
    json.dump(res, open(os.path.join(R, ("bench_el_tune" if a.phase == "tune" else "bench_el") + a.out_suffix + ".json"), "w"), indent=1, default=str)
    print("DONE", a.phase)


if __name__ == "__main__":
    main()
