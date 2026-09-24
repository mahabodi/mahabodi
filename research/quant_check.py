"""fp32 vs int8 variants of the same Laya ONNX model, through Bodi (laya-compatible maths),
on seeded samples of ag_news and emotion. Reports accuracy, agreement with fp32, and time."""
import json, os, sys, time
os.environ.setdefault("USE_TF", "0")
from datasets import load_dataset
from mahabodi import Bodi
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "laya-v2")  # int8 variants: regenerate with research/quantize.py first
ag = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0).select(range(N))
em = load_dataset("dair-ai/emotion", "split", split="test").shuffle(seed=0).select(range(N))
suites = {
  "ag_news": ([{"article": r["text"]} for r in ag], [r["label"] for r in ag], {"topic": {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": {"world": "world news and international politics", "sports": "sports", "business": "business and economy", "sci_tech": "science and technology"}}}),
  "emotion": ([{"text": r["text"]} for r in em], [r["label"] for r in em], {"emotion": {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?", "criteria": {x: None for x in ["sadness", "joy", "love", "anger", "fear", "surprise"]}}}),
}
compat = dict(permute=False, max_options_per_pass=0, script_gate=False, cache=False)
out, base = {}, {}
for mf in ["model.onnx", "model.int8pc.onnx", "model.int8pc_rr.onnx"]:
    b = Bodi(); b.load_laya(M, model_file=mf, intra_threads=8)
    b.decide_batch(suites["ag_news"][0][:4], suites["ag_news"][2], **compat)  # warm-up
    for name, (states, gold, q) in suites.items():
        qid = next(iter(q)); labels = list(q[qid]["criteria"])
        t = time.time(); res = b.decide_batch(states, q, **compat); el = time.time() - t
        pred = [labels.index(r["answers"][qid]["choice"]) for r in res]
        acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
        if mf == "model.onnx": base[name] = pred
        agree = sum(p == b_ for p, b_ in zip(pred, base[name])) / len(pred)
        out.setdefault(mf, {})[name] = {"accuracy": round(acc, 4), "agree_with_fp32": round(agree, 4), "seconds": round(el, 2), "decisions_per_s": round(len(states) / el, 2)}
        print(mf, name, out[mf][name], flush=True)
    del b
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "quant_check.json"), "w"), indent=1)
