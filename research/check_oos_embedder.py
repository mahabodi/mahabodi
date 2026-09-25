"""W11 parity step 1: MahaBodi's ONNX MiniLM (embed_text) vs the sentence-transformers max-sim saved in
bench_clinc_oos_arrays.json, on all 1,600 saved items (val + test). Reports max |delta max_sim| and
the arm-C gate flags (top-1 prob < tau OR max_sim < s) that would flip using the SAVED top-1 probs.

    .venv/bin/python research/check_oos_embedder.py
"""
import json, os
import numpy as np
from datasets import load_dataset
from mahabodi import Bodi
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
A = json.load(open(os.path.join(R, "bench_clinc_oos_arrays.json")))
th = json.load(open(os.path.join(R, "bench_clinc_oos.json")))["thresholds"]["C"]
names = load_dataset("clinc_oos", "plus", split="validation").features["intent"].names
oos = names.index("oos")
labels = [n.replace("_", " ") for i, n in enumerate(names) if i != oos]
val = load_dataset("clinc_oos", "plus", split="validation")
vrows = [r for r in val if r["intent"] == oos] + [r for r in val.shuffle(seed=7) if r["intent"] != oos][:500]
trows = list(load_dataset("clinc_oos", "plus", split="test").shuffle(seed=7).select(range(1000, 2000)))
b = Bodi(); b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
L = np.array(b.embed_text(labels), dtype=np.float64)
out = {"tau": th["tau"], "s": th["s"]}
for split, rows in (("val", vrows), ("test", trows)):
    D = A[split]
    assert [names[r["intent"]].replace("_", " ") if r["intent"] != oos else "oos" for r in rows] == D["gold"], "rows differ from saved arrays"
    U = np.array(b.embed_text([r["text"] for r in rows]), dtype=np.float64)
    ms = (U @ L.T).max(1)
    d = np.abs(ms - np.array(D["max_sim"]))
    f_st = [(c < th["tau"]) or (m < th["s"]) for (_, c), m in zip(D["C"], D["max_sim"])]
    f_mb = [(c < th["tau"]) or (m < th["s"]) for (_, c), m in zip(D["C"], ms)]
    flips = [i for i, (x, y) in enumerate(zip(f_st, f_mb)) if x != y]
    out[split] = {"n": len(rows), "max_abs_delta_max_sim": float(d.max()), "mean_abs_delta": float(d.mean()),
                  "flag_flips": len(flips), "flip_items": [{"i": i, "st": D["max_sim"][i], "onnx": float(ms[i])} for i in flips]}
    print(split, {k: v for k, v in out[split].items() if k != "flip_items"}, flush=True)
json.dump(out, open(os.path.join(R, "check_oos_embedder.json"), "w"), indent=1)
