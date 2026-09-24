"""Bodi vs Laya, same machine, same checkpoint (convaiinnovations/laya, English).

Accuracy suites are Laya's own Jev-comparable suites, built exactly as in Laya's
research/scripts/bench_apps.py (first N test rows, same instructions and criteria), and
scored with Laya's own `metrics()` from research/scripts/bench_local.py.

Systems
  laya_torch       Laya PyTorch on CPU, Laya's own batched scorer `score_cases` (throughput path)
  laya_onnx        Laya's own ONNXAgent (laya/onnx_agent.py) on the same exported model.onnx
  bodi_compat      Bodi, DecideOptions.laya_compatible(): same maths as Laya, Rust + ORT
  bodi             Bodi default: order ensemble + tournament shortlist (cache OFF)
  banking77 only:  laya_head512 (head_max_len=512) and laya_shortlist (laya.shortlist.predict_shortlist
                   with embed_fn_from_agent, k=20) - Laya's two documented mitigations.

Rules: cache off; torch.set_num_threads(T) and ORT intra_threads=T (T=8 physical cores);
warm-up excluded from latency; every decision counted; rows/decision reported.

    .venv/bin/python research/bench.py --n 400 --out research/results/bench.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYA_SRC = os.environ.get("LAYA_SRC", "/private/tmp/claude-501/-Users-pkpro-bodi/d4fcd3ac-cbb3-499b-80aa-7a890e88aabd/scratchpad/laya")
sys.path.insert(0, os.path.join(LAYA_SRC, "research", "scripts"))

from bench_local import metrics, score_cases, softmax_t, temp_for  # noqa: E402  (Laya's own)
from laya import Agent  # noqa: E402
from laya.common import QTYPES  # noqa: E402

from mahabodi import Bodi  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "laya")


def build_suites(n):
    """Verbatim construction of Laya's jev.* suites (bench_apps.py build())."""
    from datasets import load_dataset
    suites = {}
    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    keys = list(crit)
    suites["ag_news"] = ([({"article": r["text"]}, {"topic": {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": dict(crit)}})
                          for r in list(d)[:n]], [int(r["label"]) for r in list(d)[:n]], keys)
    d = load_dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    suites["emotion"] = ([({"text": r["text"]}, {"emotion": {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?",
                                                            "criteria": {x: None for x in names}}}) for r in list(d)[:n]],
                         [int(r["label"]) for r in list(d)[:n]], names)
    d = load_dataset("mteb/banking77", split="test")
    labels = sorted(set(d["label_text"]))
    lab = [x.replace("_", " ") for x in labels]
    suites["banking77"] = ([({"message": r["text"]}, {"intent": {"type": "choice", "instructions": "Which banking intent does `message` express?",
                                                                 "criteria": {x: None for x in lab}}}) for r in list(d)[:n]],
                           [lab.index(r["label_text"].replace("_", " ")) for r in list(d)[:n]], lab)
    return suites


def laya_rows(agent, cases, gold):
    t0 = time.time()
    out, index, _, dropped = score_cases(agent, cases)
    el = time.time() - t0
    rows = []
    for (ci, qid, qt, k), lg in zip(index, out):
        if lg is None:
            rows.append((gold[ci], None))
            continue
        rows.append((gold[ci], softmax_t(lg, temp_for(agent, qt, k)).tolist()))
    return rows, el, dropped


def bodi_rows(b, cases, gold, labels, opts):
    states = [c[0] for c in cases]
    qs = cases[0][1]
    qid = next(iter(qs))
    s0 = b.stats().get("system1", {})
    t0 = time.time()
    res = b.decide_batch(states, qs, **opts)
    el = time.time() - t0
    s1 = b.stats()["system1"]
    rows = [(g, [r["answers"][qid]["probabilities"][l] for l in labels]) for g, r in zip(gold, res)]
    return rows, el, s1["rows_scored"] - s0.get("rows_scored", 0)


def flip_rate(pred_a, pred_b):
    return round(float(np.mean([a != b for a, b in zip(pred_a, pred_b)])), 4)


def reversed_cases(cases):
    out = []
    for st, qs in cases:
        (qid, q), = qs.items()
        q2 = dict(q)
        q2["criteria"] = dict(reversed(list(q["criteria"].items())))
        out.append((st, {qid: q2}))
    return out


def argmax_labels(rows, labels):
    return [labels[int(np.argmax(p))] if p is not None else None for _, p in rows]


def lat(fn, cases, warm=5, n=60):
    for st, q in cases[:warm]:
        fn(st, q)
    ts = []
    for st, q in cases[warm:warm + n]:
        t = time.perf_counter()
        fn(st, q)
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1), "n": len(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "bench.json"))
    ap.add_argument("--skip", default="", help="comma list: laya_onnx,shortlist,latency,flip")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    torch.set_num_threads(a.threads)

    env = {"machine": platform.machine(), "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
           "os": platform.platform(), "threads": a.threads, "torch": torch.__version__, "python": platform.python_version(),
           "laya": __import__("laya").__version__, "onnxruntime": __import__("onnxruntime").__version__, "n_per_suite": a.n,
           "checkpoint": "convaiinnovations/laya (English, ModernBERT-large)", "date": time.strftime("%Y-%m-%d %H:%M")}
    print(json.dumps(env), flush=True)
    res = {"env": env, "suites": {}, "latency": {}}

    agent = Agent("convaiinnovations/laya", device="cpu")
    b = Bodi()
    b.load_laya(MODEL_DIR, intra_threads=a.threads)
    compat = dict(permute=False, max_options_per_pass=0, script_gate=False, cache=False)
    default = dict(cache=False)
    suites = build_suites(a.n)

    for name, (cases, gold, labels) in suites.items():
        r = {}
        rows, el, dropped = laya_rows(agent, cases, gold)
        r["laya_torch"] = {**metrics(rows), "dropped": dropped, "seconds": round(el, 1), "decisions_per_s": round(len(cases) / el, 2), "rows_per_decision": 1.0}
        laya_pred = argmax_labels(rows, labels)
        print(name, "laya_torch", r["laya_torch"], flush=True)
        for sysname, opts in (("bodi_compat", compat), ("bodi", default)):
            rows, el, nrows = bodi_rows(b, cases, gold, labels, opts)
            r[sysname] = {**metrics(rows), "seconds": round(el, 1), "decisions_per_s": round(len(cases) / el, 2), "rows_per_decision": round(nrows / len(cases), 2)}
            r[sysname + "_pred"] = argmax_labels(rows, labels)
            print(name, sysname, r[sysname], flush=True)
        r["bodi_compat_vs_laya_torch_disagreement"] = flip_rate(laya_pred, r.pop("bodi_compat_pred"))
        bodi_pred = r.pop("bodi_pred")

        if "flip" not in skip and name != "banking77":
            m = min(200, len(cases))
            rc = reversed_cases(cases[:m])
            rows_r, _, _ = laya_rows(agent, rc, gold[:m])
            # reversed criteria: labels list reversed, map back by label name
            rl = list(reversed(labels))
            r["order_flip_rate"] = {
                "laya_torch": flip_rate(laya_pred[:m], argmax_labels(rows_r, rl)),
                "bodi": flip_rate(bodi_pred[:m], [x for x in argmax_labels(bodi_rows(b, rc, gold[:m], rl, default)[0], rl)]),
                "n": m}
            print(name, "order_flip", r["order_flip_rate"], flush=True)

        if name == "banking77":
            saved = agent.cfg.get("head_max_len", 192)
            agent.cfg["head_max_len"] = 512
            rows, el, dropped = laya_rows(agent, cases, gold)
            agent.cfg["head_max_len"] = saved
            r["laya_head512"] = {**metrics(rows), "dropped": dropped, "seconds": round(el, 1), "decisions_per_s": round(len(cases) / el, 2)}
            print(name, "laya_head512", r["laya_head512"], flush=True)
            if "shortlist" not in skip:
                from laya.shortlist import embed_fn_from_agent, predict_shortlist
                ef = embed_fn_from_agent(agent)
                rows = []
                t0 = time.time()
                for (st, qs), g in zip(cases, gold):
                    out = predict_shortlist(agent, st, qs, ef, k=20)
                    probs = out["answers"]["intent"]["probabilities"]
                    rows.append((g, [probs.get(l, 0.0) for l in labels]))
                el = time.time() - t0
                r["laya_shortlist_k20"] = {**metrics(rows), "seconds": round(el, 1), "decisions_per_s": round(len(cases) / el, 2)}
                print(name, "laya_shortlist_k20", r["laya_shortlist_k20"], flush=True)
        res["suites"][name] = r
        json.dump(res, open(a.out, "w"), indent=1)

    if "latency" not in skip:
        cases = suites["ag_news"][0]
        L = res["latency"]
        L["laya_torch_predict"] = lat(lambda s, q: agent.predict(s, q), cases)
        L["bodi_compat_predict"] = lat(lambda s, q: b.predict(s, q), cases)
        L["bodi_default_decide"] = lat(lambda s, q: b.decide(s, q, cache=False), cases)
        if "laya_onnx" not in skip:
            from laya.onnx_agent import ONNXAgent
            snap = os.path.dirname(os.path.dirname(agent.tok.name_or_path)) if False else None
            from huggingface_hub import snapshot_download
            ckpt = snapshot_download("convaiinnovations/laya", allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"])
            oa = ONNXAgent(ckpt, onnx_path=os.path.join(MODEL_DIR, "model.onnx"))
            L["laya_onnx_predict"] = lat(lambda s, q: oa.predict(s, q), cases)
        print("latency", L, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
