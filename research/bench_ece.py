"""Calibration (ECE) as a SCORED row, with Laya's own protocol: "refitting one temperature ... on
held-out data" (Laya README, Calibration). Both systems get the identical refit.

Per suite:
  1. Validation items = tune_experience.suites(2000, 300)[suite]["val"], per suite:
       ag_news, banking77, boolq: train.shuffle(2)[2000:2300] (ag_news/boolq train is in Laya's
       training mix - disclosed); emotion, sst5: validation.shuffle(2)[:300];
       prompt_injections: train.shuffle(2)[:100] (100 items). Both systems score them.
  2. One temperature per system per suite, fitted by minimising NLL on validation over a grid
     (0.25..8, log-spaced). Scaling acts on each system's SUPPORT only: options with p = 0 exactly
     (tournament-eliminated in MahaBodi) stay 0; p_i ** (1/t) renormalised over p_i > 0 (= logit
     temperature scaling over the options the system kept). Items whose gold option has p = 0 get
     a fixed NLL floor (log 1e-12) independent of t; their count is reported.
  3. Test items = bench.json's 500 (bench.build_suites), scored by both; ECE (15 equal-width bins
     on top-1 confidence, as bench_local.metrics) after each system's own fitted temperature.
  4. Verdict fixed before running: paired bootstrap (2000 resamples of test items, seed 0) of
     ECE(MahaBodi) - ECE(Laya); beat if the 95% CI is below 0, loss if above 0, else tie.
Systems: laya_torch (Laya's score_cases) and MahaBodi zero-shot defaults (experience off, cache off).
Raw (unrefit) ECE is reported too. Laya's published 0.081 refits one temperature per (question type,
option count) bucket and its README does not name the suites averaged, so the mean here is reported
WITHOUT a head-to-head against 0.081.

    .venv/bin/python research/bench_ece.py --out research/results/bench_ece.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites, laya_rows, bodi_rows  # noqa: E402
from tune_experience import suites as train_suites, label_idx  # noqa: E402
from laya import Agent  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 61))


def scale(P, t):
    Q = np.where(P > 0, np.power(np.where(P > 0, P, 1.0), 1.0 / t), 0.0)
    return Q / Q.sum(1, keepdims=True)


def nll(P, Y):
    g = P[np.arange(len(Y)), Y]
    return float(-np.mean(np.where(g > 0, np.log(np.where(g > 0, g, 1.0)), np.log(1e-12))))


def ece(P, Y, bins=15):
    conf = P.max(1); corr = (P.argmax(1) == Y).astype(float)
    e = 0.0
    for lo in range(bins):
        m = (conf > lo / bins) & (conf <= (lo + 1) / bins)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def arr(rows, n):
    P = np.array([p if p is not None else [1.0 / n] * n for _, p in rows], dtype=float)
    return P, np.array([g for g, _ in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "bench_ece.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    torch.set_num_threads(8)
    only = set(filter(None, a.only.split(","))) or {"ag_news", "emotion", "banking77", "sst5", "prompt_injections", "boolq"}
    base = json.load(open(os.path.join(R, "bench.json")))["suites"]
    agent = Agent("convaiinnovations/laya", device="cpu")
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    opts = dict(cache=False, round_probabilities=False, experience_k=0)
    tr = train_suites(2000, 300)
    tests = build_suites(500, only)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"suites": {}}
    npz = a.out.replace(".json", "_probs.npz")
    arrays = dict(np.load(npz)) if os.path.exists(npz) else {}
    res["arrays"] = os.path.basename(npz) + ": <suite>__<laya_torch|bodi>__<val|test>_<P|Y>, RAW probabilities (float64, unrounded) and gold indices"
    res["protocol"] = __doc__.split("\n\n")[1]
    for name in sorted(only):
        if name in res["suites"]:
            continue
        S = tests[name]
        assert S["gold"] == base[name]["gold"], "test items differ from bench.json"
        T = tr[name]; qid, qd = next(iter(T["q"].items()))
        assert qd == S["q"] and qid == S["qid"]
        V = dict(S, states=[T["st"](x) for x in T["val"]], gold=[label_idx(qd, T["y"](x)) for x in T["val"]])
        n = len(S["labels"])
        t0 = time.time()
        out = {}
        for sys_name, fn in (("laya_torch", lambda D: laya_rows(agent, D)[0]), ("bodi", lambda D: bodi_rows(b, D, opts)[0])):
            Pv, Yv = arr(fn(V), n)
            Pt, Yt = arr(fn(S), n)
            t = float(GRID[int(np.argmin([nll(scale(Pv, g), Yv) for g in GRID]))])
            arrays["%s__%s__val_P" % (name, sys_name)] = Pv.astype(np.float64); arrays["%s__%s__val_Y" % (name, sys_name)] = Yv
            arrays["%s__%s__test_P" % (name, sys_name)] = Pt.astype(np.float64); arrays["%s__%s__test_Y" % (name, sys_name)] = Yt
            out[sys_name] = {"temperature": round(t, 4), "val_gold_eliminated": int((Pv[np.arange(len(Yv)), Yv] == 0).sum()),
                             "test_gold_eliminated": int((Pt[np.arange(len(Yt)), Yt] == 0).sum()), "val_n": len(Yv), "ece_raw": round(ece(Pt, Yt), 4), "ece_refit": round(ece(scale(Pt, t), Yt), 4),
                             "accuracy": round(float((Pt.argmax(1) == Yt).mean()), 4), "_P": scale(Pt, t), "_Y": Yt}
        rng = np.random.default_rng(0)
        Pb, Pl, Y = out["bodi"]["_P"], out["laya_torch"]["_P"], out["bodi"]["_Y"]
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(Y), len(Y))
            d.append(ece(Pb[i], Y[i]) - ece(Pl[i], Y[i]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "beat" if hi < 0 else ("loss" if lo > 0 else "tie")
        for k in out:
            out[k].pop("_P"); out[k].pop("_Y")
        out["ece_diff_bodi_minus_laya"] = round(out["bodi"]["ece_refit"] - out["laya_torch"]["ece_refit"], 4)
        out["ece_diff_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
        out["verdict"] = verdict
        out["seconds"] = round(time.time() - t0, 1)
        res["suites"][name] = out
        print(name, out, flush=True)
        np.savez_compressed(npz, **arrays)  # raw (unrefit) per-item probabilities + gold, every suite and system
        json.dump(res, open(a.out, "w"), indent=1)
    S_ = res["suites"]
    res["mean_ece_refit"] = {k: round(float(np.mean([S_[s][k]["ece_refit"] for s in S_])), 4) for k in ("laya_torch", "bodi")}
    print("mean refit ECE", res["mean_ece_refit"])
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
