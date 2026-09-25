"""W11: out-of-scope gate re-test on CLINC150 "plus" (pre-registered in research/USECASES.md,
"Pre-registered: out-of-scope gate (W11)", fixed 2026-09-25 before any run).

Arms (as bench_clinc.py): A Laya over all 150 intents; B Laya on Laya's own shortlist_choice (k=20,
all-MiniLM-L6-v2); C MahaBodi defaults (tournament). Every arm gets the SAME gate with its OWN
thresholds: answer "oos" if top-1 probability < tau OR max cosine(MiniLM(utterance), MiniLM(intent
name)) < s (names only, zero-shot).

Tuning (offline, 2-D grid): all 100 validation oos items + 500 seeded in-scope validation items;
objective = overall 151-class accuracy with oos items weighted to the documented TEST prevalence
(1000 / 5500 = 18.2 %). Test: FRESH positions 1000..1999 of the seed-7 test shuffle (asserted
disjoint from bench_clinc.json's 0..999); each arm decides once, per-item top-1 label / top-1 prob /
max-sim are saved for val and test (research/results/bench_clinc_oos_arrays.json) and thresholds
are applied offline. Win: C+gate beats B+gate on overall accuracy (exact McNemar). Also reported for
all arms: in-scope accuracy, OOS recall / precision, OOS AUROC of max-sim and of top-1 prob.

Tie-break when several (tau, s) give the same weighted validation accuracy: the smaller tau, then
the smaller s (i.e. flag LESS). If an arm's chosen tau or s is the lowest or highest enabled grid
value, the grid is extended outward (offline, on the saved arrays) and re-selected on validation,
up to 3 times; boundary flags and the grids used are recorded.

Scope of any result: the gate is applied HERE, in this script, with sentence-transformers, not by
MahaBodi's decide(). A win supports "a names-similarity gate helps", not "MahaBodi's decide()
detects OOS". If the gate is later built into decide(), that product feature must first reproduce
this script's offline gate flags exactly on the same saved items (parity check) before the README
may say MahaBodi does it.

The same max-sim gate is given to A, B and C, so OOS gains are expected in all three. OOS recall /
precision / AUROC comparisons between arms are DESCRIPTIVE only; the one pre-registered win test is
C+gate vs B+gate on overall accuracy (exact McNemar). No "OOS win" is claimed from recall differences.

    .venv/bin/python research/bench_clinc_oos.py
"""
import json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import LAYA_COMMIT, mcnemar, wilson  # noqa: E402
from datasets import load_dataset  # noqa: E402
from laya.shortlist import shortlist_choice  # noqa: E402
from mahabodi import Bodi  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
INS = "Which intent does `utterance` express?"
TEST_OOS_RATE = 1000 / 5500


def auroc(score_oos, is_oos):
    """P(score of a random oos item > score of a random in-scope item); ties count 1/2."""
    s = np.asarray(score_oos, float); y = np.asarray(is_oos, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    t0 = time.time()
    names = load_dataset("clinc_oos", "plus", split="validation").features["intent"].names
    oos = names.index("oos")
    intents = [i for i in range(len(names)) if i != oos]
    labels = [names[i].replace("_", " ") for i in intents]
    crit = {l: None for l in labels}
    q = {"intent": {"type": "choice", "instructions": INS, "criteria": crit}}
    val = load_dataset("clinc_oos", "plus", split="validation")
    vo = [r for r in val if r["intent"] == oos]
    vi = [r for r in val.shuffle(seed=7) if r["intent"] != oos][:500]
    vrows = vo + vi
    test = load_dataset("clinc_oos", "plus", split="test").shuffle(seed=7)
    prev = json.load(open(os.path.join(R, "bench_clinc.json")))
    assert [names[r["intent"]].replace("_", " ") if r["intent"] != oos else "oos" for r in test.select(range(1000))] == prev["gold"], \
        "positions 0..999 differ from bench_clinc.json"
    trows = list(test.select(range(1000, 2000)))  # fresh, disjoint from 0..999 by construction
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    ef = lambda texts: st.encode(list(texts), batch_size=64, convert_to_numpy=True)
    L = st.encode(labels, normalize_embeddings=True)
    compat = dict(permute=False, max_options_per_pass=0, script_gate=False, cache=False, experience_k=0, round_probabilities=False)

    def decide(rows):
        U = st.encode([r["text"] for r in rows], normalize_embeddings=True)
        sim = (U @ L.T).max(1).tolist()
        out = {"A": [], "B": [], "C": [], "max_sim": [round(x, 6) for x in sim],
               "gold": [names[r["intent"]].replace("_", " ") if r["intent"] != oos else "oos" for r in rows]}
        for r in rows:
            s = {"utterance": r["text"]}
            pa = b.decide(s, q, **compat)["answers"]["intent"]["probabilities"]
            out["A"].append([max(pa, key=pa.get), max(pa.values())])
            short = shortlist_choice(s, crit, ef, k=20, instructions=INS)
            pb = b.decide(s, {"intent": {"type": "choice", "instructions": INS, "criteria": {l: None for l in short}}}, **compat)["answers"]["intent"]["probabilities"]
            out["B"].append([max(pb, key=pb.get), max(pb.values())])
            pc = b.decide(s, q, cache=False, experience_k=0, round_probabilities=False)["answers"]["intent"]["probabilities"]
            out["C"].append([max(pc, key=pc.get), max(pc.values())])
        return out

    V = decide(vrows); print("val decided", round(time.time() - t0), flush=True)
    T = decide(trows); print("test decided", round(time.time() - t0), flush=True)
    arrays = {"val": V, "test": T, "val_items": "all 100 val oos + validation.shuffle(7) first 500 in-scope",
              "test_items": "test.shuffle(7) positions 1000..1999"}
    json.dump(arrays, open(os.path.join(R, "bench_clinc_oos_arrays.json"), "w"))

    def apply(D, arm, tau, s):
        return ["oos" if (c < tau or m < s) else p for (p, c), m in zip(D[arm], D["max_sim"])]

    vgold = V["gold"]; vis = np.array([g == "oos" for g in vgold])
    w = np.where(vis, TEST_OOS_RATE / vis.mean(), (1 - TEST_OOS_RATE) / (1 - vis.mean()))  # reweight to test prevalence
    res = {"laya_commit": LAYA_COMMIT, "protocol": __doc__.split("\n\n")[1:6], "thresholds": {}, "val": {}, "test": {}}
    for arm in "ABC":
        taus = [0.0] + [round(x, 3) for x in np.linspace(0.05, 0.95, 19)]
        sims = [-1.0] + [round(x, 3) for x in np.linspace(0.05, 0.60, 12)]
        for ext in range(4):
            best = max(((t, s_) for t in taus for s_ in sims),
                       key=lambda ts: (float(np.sum(w * (np.array(apply(V, arm, *ts)) == np.array(vgold)))), -ts[0], -ts[1]))
            et, es = [x for x in taus if x > 0], [x for x in sims if x > -1.0]
            tb = best[0] > 0 and best[0] in (et[0], et[-1]); sb = best[1] > -1.0 and best[1] in (es[0], es[-1])
            if not (tb or sb) or ext == 3:
                break
            if tb:
                step = et[1] - et[0]
                taus = sorted(set(taus) | ({round(et[-1] + step * k, 3) for k in (1, 2, 3) if et[-1] + step * k < 1.0} if best[0] == et[-1] else
                                            {round(et[0] / 2 ** k, 4) for k in (1, 2, 3)}))
            if sb:
                step = es[1] - es[0]
                sims = sorted(set(sims) | ({round(es[-1] + step * k, 3) for k in (1, 2, 3)} if best[1] == es[-1] else
                                            {round(es[0] - step * k, 3) for k in (1, 2, 3)}))
        res["thresholds"][arm] = {"tau": best[0], "s": best[1], "tau_on_boundary": tb, "s_on_boundary": sb, "grid_extensions": ext,
                                  "tau_grid": [taus[1], taus[-1]], "s_grid": [sims[1], sims[-1]]}
        res["val"][arm] = {"weighted_accuracy": round(float(np.sum(w * (np.array(apply(V, arm, *best)) == np.array(vgold))) / w.sum()), 4)}
    tgold = T["gold"]; tis = [g == "oos" for g in tgold]
    corr = {}
    for arm in "ABC":
        th = res["thresholds"][arm]
        pred = apply(T, arm, th["tau"], th["s"])
        c = [p == g for p, g in zip(pred, tgold)]; corr[arm] = c
        ins = [ok for ok, g in zip(c, tgold) if g != "oos"]
        tp = sum(p == "oos" and g == "oos" for p, g in zip(pred, tgold)); flagged = sum(p == "oos" for p in pred)
        res["test"][arm] = {"accuracy": round(float(np.mean(c)), 4), "accuracy_ci95": wilson(sum(c), len(c)),
                            "in_scope_accuracy": round(float(np.mean(ins)), 4), "oos_recall": round(tp / max(1, sum(tis)), 4),
                            "oos_precision": round(tp / max(1, flagged), 4), "oos_flagged": flagged, "oos_correct": tp,
                            "auroc_top1_prob": auroc([-c_ for _, c_ in T[arm]], tis), "pred": pred}
    res["test"]["auroc_max_sim"] = auroc([-m for m in T["max_sim"]], tis)
    res["mcnemar_Cgate_vs_Bgate"] = mcnemar(corr["C"], corr["B"])
    res["mcnemar_Cgate_vs_Agate"] = mcnemar(corr["C"], corr["A"])
    res["test_n"] = len(tgold); res["test_oos"] = sum(tis); res["seconds"] = round(time.time() - t0, 1)
    for arm in "ABC":
        print(arm, res["thresholds"][arm], {k: v for k, v in res["test"][arm].items() if k != "pred"}, flush=True)
    print("auroc max_sim", res["test"]["auroc_max_sim"], "C vs B", res["mcnemar_Cgate_vs_Bgate"], flush=True)
    json.dump(res, open(os.path.join(R, "bench_clinc_oos.json"), "w"))


if __name__ == "__main__":
    main()
