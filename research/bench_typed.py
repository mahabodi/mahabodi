"""typed-decisions (LocalLLaMA/typed-decisions, config "all", split "test": 400 cases / 2,000
decisions) with the laya-typed-decisions checkpoint on both sides - the checkpoint behind Laya's
published 0.766. Gold construction and every metric are copied from Laya's
research/scripts/bench_local.py (build_typed_decisions + run_part_b). The full split is used.

  laya_torch  Laya PyTorch (convaiinnovations/laya-typed-decisions), Laya's `score_cases`
  bodi        MahaBodi defaults (cache off, exact probabilities) on models/laya-typed-decisions

Per-decision correctness is compared with an exact McNemar test. Timing in this file is only
valid if nothing else ran on the machine (recorded as `concurrent_load`).

    .venv/bin/python research/bench_typed.py --out research/results/bench_typed.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import LAYA_COMMIT, LAYA_SRC, mcnemar, wilson, boot_ci  # noqa: E402
sys.path.insert(0, os.path.join(LAYA_SRC, "research", "scripts"))
from bench_local import metrics, score_cases, softmax_t, temp_for  # noqa: E402
from laya import Agent  # noqa: E402
from mahabodi import Bodi  # noqa: E402

QT = {0: "choice", 1: "score", 2: "noul"}


def build():
    from datasets import load_dataset
    td = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    cases, gold, wfs = [], [], []
    for r in td:
        qs = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        g = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        st = r["state"]
        try:
            st = json.loads(st)
        except Exception:
            pass
        gm = {}
        for qid, qd in qs.items():
            gg = g[qid]
            if qd["type"] == "choice":
                keys = list(qd["criteria"].keys())
                gm[qid] = {"idx": keys.index(str(gg["label"])), "soft": [float(gg.get("probabilities", {}).get(k, 0.0)) for k in keys]}
            elif qd["type"] == "noul":
                pt = float(gg.get("probabilities", {}).get("true", gg.get("noul", 0.5)))
                gm[qid] = {"idx": 1 if str(gg["label"]).lower() == "true" else 0, "soft": [1 - pt, pt]}
            else:
                n = len(qd["criteria"])
                gm[qid] = {"idx": int(gg["label"]), "soft": [float(gg.get("probabilities", {}).get(str(i), 0.0)) for i in range(n)],
                           "gold_score": float(gg.get("score", float(gg["label"])))}
        cases.append((st, qs)); gold.append(gm); wfs.append(r["workflow"])
    return cases, gold, wfs


def score(entries, n_cases, secs):
    """entries: list of (gold dict, probs or None, workflow, qtype name)."""
    rows, soft, brier_s, mae, w1, by_wf, by_qt = [], [], [], [], [], {}, {}
    for g, p, wf, qt in entries:
        if p is None:
            rows.append((g["idx"], None)); continue
        p = np.asarray(p, float)
        rows.append((g["idx"], p))
        by_wf.setdefault(wf, []).append((g["idx"], p)); by_qt.setdefault(qt, []).append((g["idx"], p))
        gp = np.asarray(g["soft"], float)
        if gp.sum() > 0:
            gp = gp / gp.sum()
            pp = p[:len(gp)] if len(p) >= len(gp) else np.pad(p, (0, len(gp) - len(p)))
            pp = pp / max(pp.sum(), 1e-12)
            soft.append(float((pp * gp).sum())); brier_s.append(float(((pp - gp) ** 2).sum()))
        if "gold_score" in g:
            exp = float((np.arange(len(p)) * p).sum())
            mae.append(abs(exp - g["gold_score"])); w1.append(float(abs(exp - g["gold_score"]) <= 1))
    m = metrics(rows)
    corr = [int(np.argmax(p)) == gi for gi, p in rows if p is not None]
    m.update(accuracy_ci95=wilson(sum(corr), len(corr)), ece_ci95=boot_ci(rows, "ece"),
             soft_accuracy=round(float(np.mean(soft)), 4), brier_vs_soft=round(float(np.mean(brier_s)), 4),
             score_mae=round(float(np.mean(mae)), 4), within_1_level=round(float(np.mean(w1)), 4),
             seconds=round(secs, 1), ms_per_case=round(1000 * secs / n_cases, 1),
             by_workflow={k: metrics(v) for k, v in sorted(by_wf.items())},
             by_question_type={k: metrics(v) for k, v in sorted(by_qt.items())})
    return m, [None if p is None else int(np.argmax(p)) for _, p in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--concurrent-load", default="none", help="describe other jobs running (timing validity)")
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "bench_typed.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cases, gold, wfs = build()
    res = {"laya_commit": LAYA_COMMIT, "n_cases": len(cases), "n_decisions": sum(len(q) for _, q in cases), "concurrent_load": a.concurrent_load}

    agent = Agent("convaiinnovations/laya-typed-decisions", device="cpu")
    t0 = time.time()
    lgs, idx, _, dropped = score_cases(agent, cases)
    secs = time.time() - t0
    ent = []
    for (ci, qid, qt, k), z in zip(idx, lgs):
        ent.append((gold[ci][qid], None if z is None else softmax_t(z, temp_for(agent, qt, k)), wfs[ci], QT[qt]))
    res["laya_torch"], pl = score(ent, len(cases), secs)
    res["laya_torch"]["dropped"] = dropped
    print("laya_torch", {k: v for k, v in res["laya_torch"].items() if not k.startswith("by_")}, flush=True)
    del agent

    b = Bodi()
    b.load_laya(os.path.join(ROOT, "models", "laya-typed-decisions"), intra_threads=a.threads)
    opts = {"cache": False, "round_probabilities": False}
    ent = []
    t0 = time.time()
    for ci, (st, qs) in enumerate(cases):
        out = b.decide(st, qs, **opts)["answers"]
        for qid, qd in qs.items():
            ans = out[qid]
            if ans["type"] == "noul":
                p = [1 - ans["noul"], ans["noul"]]
            elif ans["type"] == "choice":
                p = [ans["probabilities"][k] for k in qd["criteria"]]
            else:
                p = [ans["probabilities"][str(i)] for i in range(len(qd["criteria"]))]
            ent.append((gold[ci][qid], p, wfs[ci], ans["type"]))
    secs = time.time() - t0
    res["bodi"], pb = score(ent, len(cases), secs)
    print("bodi", {k: v for k, v in res["bodi"].items() if not k.startswith("by_")}, flush=True)
    golds = [g["idx"] for g, _, _, _ in ent]
    res["mcnemar_bodi_vs_laya_torch"] = mcnemar([p == g for p, g in zip(pb, golds)], [p == g for p, g in zip(pl, golds)])
    res["laya_torch_pred"], res["bodi_pred"], res["gold"] = pl, pb, golds
    print("mcnemar", res["mcnemar_bodi_vs_laya_torch"])
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
