"""Step 1 of research/PREREG_SCALE_V2.md: the store vs in-process M on the DEV 100K pool with the 500 dev mentions.

Loads the pool pages into the PostgreSQL store through the shipped engine calls, runs ONE density pass (as bench_el's
single ingest_batch), queries by mention (k = 60 -> pages -> 20, as m_short) and decides with n = 48. It compares
with bench_el_tune.json's in-process M 20_48 (per-item preds and shortlist hits).
Pass: shortlist recall within 0.03 of in-process AND accuracy an exact-McNemar tie (p >= 0.05).

    KILT_DIR=... python research/store_parity_gate.py --dsn "host=127.0.0.1 port=5433 user=postgres dbname=postgres"
"""
import argparse, json, os, re, sys, time
import numpy as np
import math, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402


# research/bench.py's wilson and mcnemar, verbatim. bench.py itself imports torch and laya at module level, which this
# store-only environment does not need, so a light stand-in module is registered before bench_el imports it.
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return {"a_only": b, "b_only": c, "p": round(min(1.0, p), 6)}


if "bench" not in sys.modules:
    sys.modules["bench"] = types.SimpleNamespace(wilson=wilson, mcnemar=mcnemar)
from bench_el import Pages, laya_choice, EL, R, ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--namespace", default="devgate")
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--skip-load", action="store_true", help="reuse an already loaded and densified namespace")
    ap.add_argument("--attempt", default="1", help="attempt label; every attempt is recorded (prereg step 1)")
    ap.add_argument("--laya", default=os.path.join(ROOT, "models", "laya-v2"))
    a = ap.parse_args()
    from mahabodi import Bodi
    t0 = time.time()
    tune = json.load(open(os.path.join(R, "bench_el_tune.json")))
    ref = tune["stages"]["100000"]["tune_M_detail"]["20_48"]
    M = [json.loads(l) for l in open(os.path.join(EL, "mentions_mq.jsonl"))]
    ms = [m for m in M if m["split"] == "dev"]
    assert [m["id"] for m in ms] == tune["mention_ids"], "dev mention order differs from bench_el_tune.json"
    gold = [m["gold_row"] for m in ms]
    pool = np.load(os.path.join(EL, "pool_dev_100000_mq.npy")).tolist()
    P = Pages()
    b = Bodi(); b.load_laya(a.laya, intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    b.call("store_open", dsn=a.dsn, namespace=a.namespace)
    times = {}
    if not a.skip_load:
        t = time.time()
        for i in range(0, len(pool), a.batch):
            rows = pool[i:i + a.batch]
            ab = P.abstracts(rows)
            b.call("store_ingest_batch", docs=[{"text": "# %s\n\n%s" % (P.titles[r], ab[r]), "source": "pg%d" % r} for r in rows])
            print("loaded", i + len(rows), "/", len(pool), round(time.time() - t, 1), "s", flush=True)
        times["load_s"] = round(time.time() - t, 1)
        t = time.time(); b.call("store_build_index"); times["build_s"] = round(time.time() - t, 1)
        t = time.time(); dens = b.call("store_ensure_density"); times["density_s"] = round(time.time() - t, 1)
        print("density", json.dumps(dens)[:400], times, flush=True)
    else:
        dens = None
    preds, hits, lat, shortlists = [], [], [], []
    for i, m in enumerate(ms):
        t = time.perf_counter()
        r = b.call("store_query", q=m["mention"] or m["state"], k=60)
        rows = []
        for h in r.get("hits", []):
            mm = re.match(r"F_pg_(\d+)_", h["id"])
            if mm and int(mm.group(1)) not in rows:
                rows.append(int(mm.group(1)))
        sh = rows[:20]
        preds.append(laya_choice(b, m["state"], sh, P, 48, decide=True))
        lat.append((time.perf_counter() - t) * 1000)
        hits.append(m["gold_row"] in sh); shortlists.append(sh)
        if i % 50 == 0:
            print("query", i, "/", len(ms), flush=True)
    acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
    rec = float(np.mean(hits))
    mc = mcnemar([p == g for p, g in zip(preds, gold)], [p == g for p, g in zip(ref["pred"], gold)])
    res = {"step": "PREREG_SCALE_V2 step 1 (dev parity gate)", "attempt": a.attempt, "provenance": provenance(), "namespace": a.namespace,
           "n": len(ms), "mention_ids": [m["id"] for m in ms], "gold_rows": gold, "store": {"accuracy": round(acc, 4), "ci95": wilson(sum(p == g for p, g in zip(preds, gold)), len(gold)),
                                   "shortlist_recall": round(rec, 4), "latency_ms_p50": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95))},
           "in_process": {"accuracy": ref["accuracy"], "shortlist_recall": ref["shortlist_recall"]},
           "mcnemar_store_vs_in_process": mc,
           "pred_agreement": int(sum(p == q for p, q in zip(preds, ref["pred"]))),
           "shortlist_hit_agreement": int(sum(x == y for x, y in zip(hits, ref["shortlist_hit"]))),
           "times": times, "density": dens, "pred": preds, "shortlist_hit": hits, "shortlists": shortlists}
    res["gate_pass"] = abs(rec - ref["shortlist_recall"]) <= 0.03 and mc["p"] >= 0.05
    json.dump(res, open(os.path.join(R, "store_parity_gate_attempt%s.json" % a.attempt), "w"), indent=1, default=str)
    print("GATE", {k: res[k] for k in ("store", "in_process", "mcnemar_store_vs_in_process", "pred_agreement", "gate_pass")}, round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
