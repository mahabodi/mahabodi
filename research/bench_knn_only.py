"""Memory-alone baseline for the experience-memory results: a kNN classifier over the SAME
memory (train examples), the SAME embedder (MahaBodi's MiniLM, via embed_text on plain_text
states) and the SAME k / temperature as each MahaBodi+experience setting - but no Laya.
Same test items as bench.json. McNemar vs laya_torch and vs MahaBodi+experience.

If knn_only >= MahaBodi+experience, the gain is the kNN memory's, not the Laya+memory fusion's.

    .venv/bin/python research/bench_knn_only.py --experience bench_experience_trust.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites, mcnemar, wilson  # noqa: E402
from tune_experience import suites as train_suites, label_idx, plain_text  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experience", default="bench_experience_trust.json")
    ap.add_argument("--out", default=os.path.join(R, "bench_knn_only.json"))
    a = ap.parse_args()
    base = json.load(open(os.path.join(R, "bench.json")))["suites"]
    exp = json.load(open(os.path.join(R, a.experience)))
    b = Bodi(); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    tr = train_suites(exp["memory_per_suite"], 300)
    S_all = build_suites(500, set(exp["suites"]))
    res = {"experience_file": a.experience, "suites": {}}
    for name, r in exp["suites"].items():
        S, T = S_all[name], tr[name]
        qid, qd = next(iter(T["q"].items()))
        n = 2 if qd["type"] == "noul" else len(qd["criteria"])
        Mv = np.array(b.embed_text([plain_text(T["st"](x)) for x in T["mem"]]), dtype=np.float32)
        My = np.array([label_idx(qd, T["y"](x)) for x in T["mem"]])
        E = np.array(b.embed_text([plain_text(s) for s in S["states"]]), dtype=np.float32)
        sims = E @ Mv.T
        gold = S["gold"]
        lp = base[name]["laya_torch_pred"]
        out = {}
        own = json.load(open(os.path.join(R, "tune_knn_only.json")))[name]
        for tag in ("per_suite", "global", "own_tuned"):
            m = r.get("bodi_experience_" + ("per_suite" if tag == "own_tuned" else tag))
            if not m:
                continue
            # own_tuned: the baseline's own validation optimum (tune_knn_only.py), compared
            # against MahaBodi's per-suite fusion
            k, temp = (own["k"], own["temperature"]) if tag == "own_tuned" else (m["settings"]["k"], m["settings"]["temperature"])
            top = np.argsort(-sims, 1)[:, :k]
            pred = []
            for i in range(len(E)):
                w = np.exp((sims[i, top[i]] - sims[i, top[i]].max()) / temp)
                pred.append(int(np.bincount(My[top[i]], weights=w, minlength=n).argmax()))
            corr = [p == g for p, g in zip(pred, gold)]
            ce = [p == g for p, g in zip(m["pred"], gold)]
            out[tag] = {"k": k, "temperature": temp, "accuracy": round(float(np.mean(corr)), 4), "accuracy_ci95": wilson(sum(corr), len(corr)),
                        "mcnemar_bodi_experience_vs_knn_only": mcnemar(ce, corr),
                        "mcnemar_knn_only_vs_laya_torch": mcnemar(corr, [p == g for p, g in zip(lp, gold)]),
                        "bodi_experience_accuracy": m["accuracy"], "pred": pred}
            print(name, tag, {k2: v for k2, v in out[tag].items() if k2 != "pred"}, flush=True)
        res["suites"][name] = out
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
