"""Alias prior + context decider on unseen KILT entity-linking sets (research/PREREG_ALIAS_PRIOR.md).

Per item, the models run once and their per-option probabilities are stored; the (tau, lambda) grid is then applied to
the stored probabilities, so tuning never re-runs a model and the test applies the frozen setting to the same kind of
record.

  P0    top alias entity (AIDA-train counts), else exact normalised title match, else no answer (as bench_el)
  L1    Laya `predict` on the dense MiniLM top-20 over all 5.9M pages (mention query, n = 16, as selected in bench_el)
  P0L   P0, else L1 (primary control)
  CTX   MahaBodi `decide` over C(m) (context only)
  AP    if max share >= tau: top alias entity; else argmax log max(p_decide, 1e-6) + lam * log(share + 0.01)
  APL   the same with Laya `predict` probabilities
C(m): alias entities in count order up to 20, filled from the dense top-k (duplicates skipped); options in a seeded
shuffled order (crc32(item id)); descriptions n = 48.

    .venv/bin/python research/bench_alias_prior.py --phase tune   -> research/results/alias_prior_tune.json
    .venv/bin/python research/bench_alias_prior.py --phase test   -> research/results/alias_prior_test.json
"""
import argparse, json, math, os, random, re, sys, time, zlib
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
from bench import wilson, mcnemar  # noqa: E402
from bench_el import Pages, option_text, ntitle, QUESTION, K, EL, R, ROOT  # noqa: E402
from kilt_el_prep import state_of  # noqa: E402

SETS = {"wned": "wned-dev-kilt.jsonl", "cweb": "cweb-dev-kilt.jsonl"}
TAUS = [0.5, 0.7, 0.9, None]          # None = never gate
LAMS = [0.0, 0.5, 1.0, 2.0]
KC, N_DESC, N_L1, FLOOR = 20, 48, 16, 1e-6


def load_set(name, row_of):
    """All eligible items in file order, shuffled with seed 0; first 200 = tune split, next <= 1000 = test split."""
    rows, dropped = [], 0
    for line in open(os.path.join(K, SETS[name])):
        d = json.loads(line)
        gid = d["output"][0]["provenance"][0]["wikipedia_id"] if d.get("output") and d["output"][0].get("provenance") else None
        if gid is None or str(gid) not in row_of:
            dropped += 1; continue
        inp = d["input"]
        men = (d.get("meta") or {}).get("mention")
        if not men and "[START_ENT]" in inp and "[END_ENT]" in inp:
            men = inp[inp.find("[START_ENT]") + len("[START_ENT]"):inp.find("[END_ENT]")].strip()
        rows.append({"id": "%s:%s" % (name, d["id"]), "mention": men or "", "state": state_of(inp), "gold_row": row_of[str(gid)]})
    random.Random(0).shuffle(rows)
    return rows[:200], rows[200:1200], {"eligible": len(rows), "dropped_gold_not_in_pages": dropped}


def choice_probs(b, state, rows, P, ntok, decide):
    """Per-row probabilities from MahaBodi `decide` or Laya `predict`; a duplicate title keeps its first row."""
    rows = list(dict.fromkeys(int(r) for r in rows))
    if not rows:
        return {}
    ab = P.abstracts(rows)
    keys, crit = {}, {}
    for r in rows:
        t = P.titles[r]
        if t in crit:
            continue
        keys[t] = r; crit[t] = option_text(t, ab[r], ntok)
    q = {"entity": {"type": "choice", "instructions": QUESTION, "criteria": crit}}
    out = b.decide({"text": state}, q) if decide else b.predict({"text": state}, q)
    probs = out["answers"]["entity"].get("probabilities") or {}
    return {keys[t]: float(p) for t, p in probs.items() if t in keys}


def combine(cands, shares, probs, tau, lam):
    if not cands:
        return None
    top_share = max(shares.values()) if shares else 0.0
    if tau is not None and top_share >= tau:
        return max(cands, key=lambda r: (shares.get(r, 0.0), -cands.index(r)))
    return max(cands, key=lambda r: (math.log(max(probs.get(r, 0.0), FLOOR)) + lam * math.log(shares.get(r, 0.0) + 0.01), -cands.index(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tune", "test"], required=True)
    ap.add_argument("--laya", default=os.path.join(ROOT, "models", "laya-v2"))
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: first N items per split (not for reporting)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--apt-only", action="store_true", help="tune only the secondary AP+title arms (clarification 2) on the tuning splits")
    a = ap.parse_args()
    from mahabodi import Bodi
    from sentence_transformers import SentenceTransformer
    t_start = time.time()
    P = Pages()
    ids = np.load(os.path.join(K, "dense", "ids.npy")); row_of = {str(x): i for i, x in enumerate(ids)}
    title_rows = defaultdict(list)
    for i, t in enumerate(P.titles):
        title_rows[ntitle(t)].append(i)
    # AIDA-train alias counts (the P0 table); for the descriptive AIDA grid the bench_el dev mentions are excluded
    M_el = [json.loads(l) for l in open(os.path.join(EL, "mentions_mq.jsonl"))]
    aida_dev = [m for m in M_el if m["split"] == "dev"]
    dev_ids = {m["id"] for m in aida_dev}
    prior_all, prior_nodev = defaultdict(Counter), defaultdict(Counter)
    for line in open(os.path.join(K, "aidayago2-train-kilt.jsonl")):
        d = json.loads(line)
        if not d["output"] or not d["output"][0].get("provenance"):
            continue
        g = str(d["output"][0]["provenance"][0]["wikipedia_id"])
        men = (d["meta"].get("mention") or "").lower().strip()
        if g in row_of and men:
            prior_all[men][row_of[g]] += 1
            if d["id"] not in dev_ids:
                prior_nodev[men][row_of[g]] += 1
    # items for this phase
    splits, meta = {}, {}
    for name in SETS:
        tune, test, info = load_set(name, row_of)
        meta[name] = dict(info, tune=len(tune), test=len(test))
        splits[name] = tune if a.phase == "tune" else test   # the other split is never scored in this phase
    if a.phase == "tune" and not a.apt_only:
        splits["aida_dev"] = [{"id": m["id"], "mention": m["mention"] or "", "state": m["state"], "gold_row": m["gold_row"], "dense_top": m["dense_top"]} for m in aida_dev]
    if a.limit:
        splits = {k: v[:a.limit] for k, v in splits.items()}
    # dense top-20 over all pages for items without a stored one (mention query, exact inner product)
    need = [m for v in splits.values() for m in v if "dense_top" not in m]
    if need:
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"); st.max_seq_length = 256
        Q = st.encode([m["mention"] or m["state"] for m in need], batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        E = np.load(os.path.join(K, "dense", "emb.f16.npy"), mmap_mode="r")
        best_s = np.full((len(need), KC), -1e4, dtype=np.float32); best_i = np.zeros((len(need), KC), dtype=np.int64)
        for s in range(0, E.shape[0], 250_000):
            sc = Q @ np.asarray(E[s:s + 250_000], dtype=np.float32).T
            top = np.argpartition(-sc, KC, axis=1)[:, :KC]
            cs = np.concatenate([best_s, np.take_along_axis(sc, top, 1)], 1); ci = np.concatenate([best_i, top + s], 1)
            j = np.argsort(-cs, axis=1)[:, :KC]
            best_s, best_i = np.take_along_axis(cs, j, 1), np.take_along_axis(ci, j, 1)
        for m, row in zip(need, best_i):
            m["dense_top"] = row.tolist()
    b = Bodi(); b.load_laya(a.laya, intra_threads=8)
    lb = Bodi(); lb.load_laya(a.laya, intra_threads=8)
    out = {"phase": a.phase, "provenance": provenance(), "sets": meta, "grid": {"tau": TAUS, "lambda": LAMS},
           "prereg": "research/PREREG_ALIAS_PRIOR.md", "results": {}}
    sel = json.load(open(os.path.join(R, "alias_prior_tune.json")))["selection"] if a.phase == "test" else None
    sel_t = json.load(open(os.path.join(R, "alias_prior_tune_apt.json")))["selection"] if a.phase == "test" else None
    if a.phase == "tune":
        out["selection"] = {}
    for name, items in splits.items():
        prior = prior_nodev if name == "aida_dev" else prior_all
        recs = []
        for i, m in enumerate(items):
            men = m["mention"].lower().strip()
            al = [r for r, _ in prior.get(men, Counter()).most_common()]
            tot = sum(prior[men].values()) if al else 0
            shares = {r: prior[men][r] / tot for r in al[:KC]} if al else {}
            cands = al[:KC]
            if len(al) < 2:
                cands += [r for r in m["dense_top"] if r not in cands][:KC - len(cands)]
            if al:
                p0 = al[0]
            else:
                tm = title_rows.get(ntitle(m["mention"]), [])
                p0 = tm[0] if tm else None
            # clarification 2 (secondary AP+title): title-match rows first, then alias entities, then the dense fill as AP;
            # unseen mentions give each title-match row share 1/#rows
            tm_all = title_rows.get(ntitle(m["mention"]), [])
            cands_t = list(dict.fromkeys(tm_all + al))[:KC]
            if len(al) < 2:
                cands_t += [r for r in m["dense_top"] if r not in cands_t][:KC - len(cands_t)]
            shares_t = dict(shares) if al else {r: 1.0 / len(tm_all) for r in tm_all[:KC]}
            order_t = list(cands_t); random.Random(zlib.crc32(m["id"].encode())).shuffle(order_t)
            rec = {"id": m["id"], "gold": m["gold_row"], "in_table": bool(al), "P0": p0,
                   "cands_t": cands_t, "shares_t": {str(k): v for k, v in shares_t.items()},
                   "ctx_t": {str(k): v for k, v in choice_probs(b, m["state"], order_t, P, N_DESC, decide=True).items()},
                   "apl_t": {str(k): v for k, v in choice_probs(lb, m["state"], order_t, P, N_DESC, decide=False).items()}}
            if not a.apt_only:
                order = list(cands); random.Random(zlib.crc32(m["id"].encode())).shuffle(order)
                ctx = choice_probs(b, m["state"], order, P, N_DESC, decide=True)
                apl = choice_probs(lb, m["state"], order, P, N_DESC, decide=False)
                l1p = choice_probs(lb, m["state"], m["dense_top"][:KC], P, N_L1, decide=False)
                l1 = max(l1p, key=l1p.get) if l1p else None
                rec.update({"cands": cands, "shares": {str(k): v for k, v in shares.items()},
                            "gold_from_fill_only": m["gold_row"] in cands and m["gold_row"] not in al,
                            "ctx": {str(k): v for k, v in ctx.items()}, "apl": {str(k): v for k, v in apl.items()},
                            "L1": l1, "P0L": p0 if p0 is not None else l1})
            recs.append(rec)
            if i % 50 == 0:
                print(a.phase, name, i, "/", len(items), flush=True)
        gold = [r["gold"] for r in recs]
        def preds(arm, tau, lam):
            c_, s_k, p_k = {"AP": ("cands", "shares", "ctx"), "APL": ("cands", "shares", "apl"),
                            "APT": ("cands_t", "shares_t", "ctx_t"), "APLT": ("cands_t", "shares_t", "apl_t")}[arm]
            return [combine(r[c_], {int(k): v for k, v in r[s_k].items()}, {int(k): v for k, v in r[p_k].items()}, tau, lam) for r in recs]
        acc = lambda p: round(float(np.mean([x == g for x, g in zip(p, gold)])), 4) if gold else None
        res = {"n": len(recs), "records": recs,
               "coverage_in_alias_table": round(float(np.mean([r["in_table"] for r in recs])), 4) if recs else None,
               "candidate_recall_t": round(float(np.mean([r["gold"] in r["cands_t"] for r in recs])), 4) if recs else None}
        if not a.apt_only:
            res["candidate_recall"] = round(float(np.mean([r["gold"] in r["cands"] for r in recs])), 4) if recs else None
            res["gold_from_fill_only"] = round(float(np.mean([r["gold_from_fill_only"] for r in recs])), 4) if recs else None
        base = {"P0": [r["P0"] for r in recs]}
        if not a.apt_only:
            base.update({"L1": [r["L1"] for r in recs], "P0L": [r["P0L"] for r in recs], "CTX": preds("AP", None, 0.0)})
        tune_arms = ("APT", "APLT") if a.apt_only else ("AP", "APL")
        if a.phase == "tune":
            tables = {}
            for arm in tune_arms:
                tables[arm] = {"%s_%s" % (t, l): acc(preds(arm, t, l)) for t in TAUS for l in LAMS}
            res["grid_accuracy"] = tables
            res["baselines"] = {k: acc(v) for k, v in base.items()}
            if name != "aida_dev":
                # pre-registered rule, as clarified in PREREG_ALIAS_PRIOR.md clarification 1: max accuracy; ties to the more
                # prior-like setting (lower tau, then larger lambda)
                out["selection"][name] = {}
                for arm in tune_arms:
                    tab = tables[arm]
                    def rank(k, tab=tab):
                        t, l = k.split("_")
                        return (tab[k], -TAUS.index(None if t == "None" else float(t)), float(l))
                    best = max(tab, key=rank)
                    t, l = best.split("_")
                    out["selection"][name][arm] = {"tau": None if t == "None" else float(t), "lambda": float(l), "tune_accuracy": tab[best]}
        else:
            s_, st_ = sel[name], sel_t[name]
            arms = dict(base, AP=preds("AP", s_["AP"]["tau"], s_["AP"]["lambda"]), APL=preds("APL", s_["APL"]["tau"], s_["APL"]["lambda"]),
                        APT=preds("APT", st_["APT"]["tau"], st_["APT"]["lambda"]), APLT=preds("APLT", st_["APLT"]["tau"], st_["APLT"]["lambda"]))
            res["arms"] = {k: {"accuracy": acc(v), "ci95": wilson(sum(x == g for x, g in zip(v, gold)), len(gold)),
                               "answered": sum(x is not None for x in v), "pred": v} for k, v in arms.items()}
            cmp_ = [("AP", "P0L"), ("AP", "P0"), ("AP", "APL"), ("AP", "CTX"), ("AP", "L1"), ("P0L", "L1"),
                    ("APT", "P0L"), ("APT", "AP"), ("APT", "APLT")]  # the last three are the secondary arms (clarification 2)
            res["mcnemar"] = {"%s_vs_%s" % (x, y): mcnemar([p == g for p, g in zip(arms[x], gold)], [p == g for p, g in zip(arms[y], gold)]) for x, y in cmp_}
            subs = {"confident": lambda r: r["in_table"] and max(r["shares"].values()) >= 0.9,
                    "ambiguous": lambda r: r["in_table"] and max(r["shares"].values()) < 0.9,
                    "unseen": lambda r: not r["in_table"]}
            def sub_acc(v, f):
                pairs = [(p, r["gold"]) for p, r in zip(v, recs) if f(r)]
                return round(float(np.mean([p == g for p, g in pairs])), 4) if pairs else None
            res["subsets"] = {sn: dict({"n": sum(f(r) for r in recs)}, **{k: sub_acc(v, f) for k, v in arms.items()}) for sn, f in subs.items()}
        out["results"][name] = res
    out["seconds"] = round(time.time() - t_start, 1)
    fname = "alias_prior_%s%s%s.json" % (a.phase, "_apt" if a.apt_only else "", a.out_suffix)
    json.dump(out, open(os.path.join(R, fname), "w"), indent=1, default=str)
    print("DONE", a.phase, {k: v.get("baselines", {k2: v2["accuracy"] for k2, v2 in v.get("arms", {}).items()}) for k, v in out["results"].items()})


if __name__ == "__main__":
    main()
