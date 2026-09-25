"""W11 parity step 2: the PRODUCT gate. MahaBodi decide() with arm C's settings (defaults, tournament)
plus oos_min_similarity / oos_below_probability = the thresholds tuned in bench_clinc_oos.json, on the
same 1,600 saved items (val + test). Parity = identical out-of-scope flags on every item; top-1
probabilities must match the saved arm-C values within float tolerance, and max_similarity must match
the saved max_sim. Product metrics on the test items are recomputed from decide()'s own flags.
No threshold is re-tuned.

    .venv/bin/python research/check_oos_product.py
"""
import hashlib, json, os, sys
os.environ.setdefault("USE_TF", "0")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import mcnemar, wilson  # noqa: E402
from datasets import load_dataset  # noqa: E402
from mahabodi import Bodi  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
A = json.load(open(os.path.join(R, "bench_clinc_oos_arrays.json")))
res0 = json.load(open(os.path.join(R, "bench_clinc_oos.json")))
th = res0["thresholds"]["C"]
names = load_dataset("clinc_oos", "plus", split="validation").features["intent"].names
oos = names.index("oos")
labels = [n.replace("_", " ") for i, n in enumerate(names) if i != oos]
q = {"intent": {"type": "choice", "instructions": "Which intent does `utterance` express?", "criteria": {l: None for l in labels}}}
val = load_dataset("clinc_oos", "plus", split="validation")
vrows = [r for r in val if r["intent"] == oos] + [r for r in val.shuffle(seed=7) if r["intent"] != oos][:500]
trows = list(load_dataset("clinc_oos", "plus", split="test").shuffle(seed=7).select(range(1000, 2000)))
b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
assert b.decide_defaults()["oos_min_similarity"] is None, "gate must be off by default"
opts = dict(cache=False, experience_k=0, round_probabilities=False, oos_min_similarity=th["s"], oos_below_probability=th["tau"])
import mahabodi, time  # noqa: E402
so = [os.path.join(os.path.dirname(mahabodi.__file__), f) for f in os.listdir(os.path.dirname(mahabodi.__file__)) if f.endswith(".so")]
out = {"thresholds": th, "provenance": {"decide_defaults": b.decide_defaults(), "module": mahabodi.__file__,
       "native_sha256": {p_: hashlib.sha256(open(p_, "rb").read()).hexdigest() for p_ in so}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}}
print("provenance", json.dumps(out["provenance"]["native_sha256"]), flush=True)
for split, rows in (("val", vrows), ("test", trows)):
    D = A[split]
    gold = [names[r["intent"]].replace("_", " ") if r["intent"] != oos else "oos" for r in rows]
    assert gold == D["gold"], "rows differ from saved arrays"
    texts_sha = hashlib.sha256("\n".join(r["text"] for r in rows).encode()).hexdigest()
    flags, top, top_p, sims = [], [], [], []
    for r in rows:
        a = b.decide({"utterance": r["text"]}, q, **opts)["answers"]["intent"]
        p = a["probabilities"]; lab = max(p, key=p.get)
        top.append(lab); top_p.append(p[lab]); sims.append(a["bodi"]["max_similarity"]); flags.append(bool(a["bodi"]["out_of_scope"]))
    ref = [(c < th["tau"]) or (m < th["s"]) for (_, c), m in zip(D["C"], D["max_sim"])]
    diff = [i for i, (x, y) in enumerate(zip(flags, ref)) if x != y]
    dp = float(np.max(np.abs(np.array(top_p) - np.array([c for _, c in D["C"]]))))
    ds = float(np.max(np.abs(np.array(sims) - np.array(D["max_sim"]))))
    lab_diff = sum(x != y for x, y in zip(top, [l for l, _ in D["C"]]))
    out[split] = {"n": len(rows), "texts_sha256": texts_sha, "flags_differ": len(diff), "differ_items": diff,
                  "max_abs_delta_top1_prob": dp, "max_abs_delta_max_sim": ds, "top1_label_differs": lab_diff,
                  "flags": flags, "top1": top}
    if split == "test":
        pred = ["oos" if f else l for f, l in zip(flags, top)]
        c = [p_ == g for p_, g in zip(pred, gold)]; ins = [ok for ok, g in zip(c, gold) if g != "oos"]
        tp = sum(p_ == "oos" and g == "oos" for p_, g in zip(pred, gold)); n_oos = sum(g == "oos" for g in gold)
        bc = [p_ == g for p_, g in zip(res0["test"]["B"]["pred"], gold)]
        out["test_product_metrics"] = {"accuracy": round(float(np.mean(c)), 4), "accuracy_ci95": wilson(sum(c), len(c)),
                                       "in_scope_accuracy": round(float(np.mean(ins)), 4), "oos_recall": round(tp / n_oos, 4),
                                       "oos_precision": round(tp / max(1, sum(p_ == "oos" for p_ in pred)), 4),
                                       "mcnemar_product_vs_Bgate": mcnemar(c, bc)}
    print(split, {k: v for k, v in out[split].items() if k not in ("flags", "top1", "differ_items")}, flush=True)
print("product test", out.get("test_product_metrics"))
json.dump(out, open(os.path.join(R, "check_oos_product.json"), "w"), indent=1)
