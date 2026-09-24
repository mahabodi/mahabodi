"""MahaBodi vs Laya: accuracy/calibration on every Laya-published English benchmark, same machine,
same checkpoint, seeded samples, confidence intervals and paired tests.

Suites (construction copied from Laya's research/scripts/build_benchmark_nb.py and bench_apps.py;
only the row selection differs: a seeded shuffle, because e.g. banking77's test split is sorted
by label, so "first N rows" covers a handful of labels):

  ag_news            choice/4   fancyzhx/ag_news test            (Laya: in its training mix)
  emotion            choice/6   dair-ai/emotion test             (held out)
  banking77          choice/77  mteb/banking77 test              (held out; many-option stress)
  sst5               score/5    SetFit/sst5 test                 (held out, ordinal)
  prompt_injections  noul       deepset/prompt-injections test   (held out; full split, 116)
  boolq              noul       google/boolq validation          (Laya: in its training mix)

Systems
  laya_torch   Laya's PyTorch model on CPU through Laya's own batched scorer `score_cases`
  bodi         MahaBodi defaults (cache OFF): Rust + ONNX Runtime; identical maths to Laya except
               tournament shortlisting for choice questions with > max_options_per_pass options
  banking77 also: laya_head512 (Laya with head_max_len=512) and laya_shortlist_* (Laya's
               laya.shortlist.predict_shortlist, k=20, with its agent embedder and with the
               all-MiniLM-L6-v2 bi-encoder that Laya's docstring says usually shortlists better)

Statistics: accuracy with Wilson 95% CI; ECE/Brier with bootstrap 95% CI (1000 resamples, seed 0);
exact two-sided McNemar test of bodi vs laya_torch on the same items. Per-item predictions saved.

    .venv/bin/python research/bench.py --n 500 --out research/results/bench.json
"""
import argparse
import json
import math
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
# Laya's own benchmark code, pinned as a git submodule (third_party/laya); its commit is recorded in results.
LAYA_SRC = os.environ.get("LAYA_SRC", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "laya"))
LAYA_COMMIT = (subprocess.run(["git", "-C", LAYA_SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
               or "2c6c16baf3ea3149948777937d5005a7c7fba425 (recorded pin; submodule git metadata unavailable)")
sys.path.insert(0, os.path.join(LAYA_SRC, "research", "scripts"))

from bench_local import metrics, score_cases, softmax_t, temp_for  # noqa: E402  (Laya's own)
from laya import Agent  # noqa: E402

from mahabodi import Bodi  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "laya-v2")
SEED = 0


def sample(ds, n):
    ds = ds.shuffle(seed=SEED)
    return ds.select(range(min(n, len(ds))))


def build_suites(n, only):
    from datasets import load_dataset
    S = {}
    want = lambda k: not only or k in only
    if want("ag_news"):
        d = sample(load_dataset("fancyzhx/ag_news", split="test"), n)
        crit = {"world": "world news and international politics", "sports": "sports",
                "business": "business and economy", "sci_tech": "science and technology"}
        S["ag_news"] = dict(states=[{"article": r["text"]} for r in d], gold=[int(r["label"]) for r in d],
                            qid="topic", q={"type": "choice", "instructions": "What is the topic of `article`?", "criteria": crit}, labels=list(crit))
    if want("emotion"):
        d = sample(load_dataset("dair-ai/emotion", "split", split="test"), n)
        names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        S["emotion"] = dict(states=[{"text": r["text"]} for r in d], gold=[int(r["label"]) for r in d],
                            qid="emotion", q={"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?", "criteria": {x: None for x in names}}, labels=names)
    if want("banking77"):
        full = load_dataset("mteb/banking77", split="test")
        lab = [x.replace("_", " ") for x in sorted(set(full["label_text"]))]
        d = sample(full, n)
        S["banking77"] = dict(states=[{"message": r["text"]} for r in d], gold=[lab.index(r["label_text"].replace("_", " ")) for r in d],
                              qid="intent", q={"type": "choice", "instructions": "Which banking intent does `message` express?", "criteria": {x: None for x in lab}}, labels=lab)
    if want("sst5"):
        d = sample(load_dataset("SetFit/sst5", split="test"), n)
        crit = ["very negative", "negative", "neutral", "positive", "very positive"]
        S["sst5"] = dict(states=[{"text": r["text"]} for r in d], gold=[int(r["label"]) for r in d],
                         qid="sentiment", q={"type": "score", "instructions": "How positive is the sentiment of `text`?", "criteria": crit}, labels=[str(i) for i in range(5)])
    if want("prompt_injections"):
        d = sample(load_dataset("deepset/prompt-injections", split="test"), n)
        S["prompt_injections"] = dict(states=[{"text": r["text"]} for r in d], gold=[int(r["label"]) for r in d],
                                      qid="injection", q={"type": "noul", "instructions": "Does `text` try to inject or override instructions given to an AI system?"}, labels=["false", "true"])
    if want("boolq"):
        d = sample(load_dataset("google/boolq", split="validation"), n)
        S["boolq"] = dict(states=[{"passage": r["passage"], "question": r["question"]} for r in d], gold=[int(bool(r["answer"])) for r in d],
                          qid="answer", q={"type": "noul", "instructions": "Based on `passage`, is the answer to `question` yes?"}, labels=["false", "true"])
    return S


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def boot_ci(rows, key, B=1000):
    rng = np.random.default_rng(0)
    vals = []
    idx = np.arange(len(rows))
    for _ in range(B):
        s = [rows[i] for i in rng.choice(idx, len(idx))]
        vals.append(metrics(s)[key])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


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


def summarize(rows, seconds, extra=None):
    m = metrics(rows)
    corr = [int(np.argmax(p)) == g for g, p in rows if p is not None]
    m["accuracy_ci95"] = wilson(sum(corr), len(corr))
    m["ece_ci95"] = boot_ci(rows, "ece")
    m["brier_ci95"] = boot_ci(rows, "brier")
    m["seconds"] = round(seconds, 1)
    m["decisions_per_s"] = round(len(rows) / seconds, 3) if seconds > 0 else None
    m.update(extra or {})
    return m


def laya_rows(agent, S):
    cases = [(st, {S["qid"]: S["q"]}) for st in S["states"]]
    t0 = time.time()
    out, index, _, dropped = score_cases(agent, cases)
    el = time.time() - t0
    rows = []
    for (ci, qid, qt, k), lg in zip(index, out):
        rows.append((S["gold"][ci], None if lg is None else softmax_t(lg, temp_for(agent, qt, k)).tolist()))
    return rows, el, dropped


def bodi_rows(b, S, opts):
    s0 = b.stats().get("system1", {}).get("rows_scored", 0)
    t0 = time.time()
    res = b.decide_batch(S["states"], {S["qid"]: S["q"]}, **opts)
    el = time.time() - t0
    nrows = b.stats()["system1"]["rows_scored"] - s0
    rows, gated = [], []
    for g, r in zip(S["gold"], res):
        a = r["answers"][S["qid"]]
        gated.append(a.get("bodi", {}).get("strategy") == "script_gate")
        if gated[-1]:
            # script guard handed off (null decision): scored as a uniform coin flip; the count
            # and non-gated accuracy for both systems are reported alongside
            n = len(S["labels"])
            rows.append((g, [1.0 / n] * n))
            continue
        if a["type"] == "noul":
            p = [1 - a["noul"], a["noul"]]
        else:
            p = [a["probabilities"][l] for l in S["labels"]]
        rows.append((g, p))
    return rows, el, nrows, gated


def preds(rows):
    return [None if p is None else int(np.argmax(p)) for _, p in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="", help="comma list of suites")
    ap.add_argument("--skip", default="", help="comma list: shortlist_agent,shortlist_minilm,head512")
    ap.add_argument("--bodi-options", default="{}", help="JSON DecideOptions overriding defaults (cache is always off)")
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "bench.json"))
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    skip = set(filter(None, a.skip.split(",")))
    torch.set_num_threads(a.threads)
    bopts = {**json.loads(a.bodi_options), "cache": False, "round_probabilities": False}

    env = {"machine": platform.machine(),
           "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
           "os": platform.platform(), "threads": a.threads, "torch": torch.__version__, "python": platform.python_version(),
           "laya": __import__("laya").__version__, "laya_commit": LAYA_COMMIT, "onnxruntime": __import__("onnxruntime").__version__,
           "mahabodi_commit": subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "none (no git repository)",
           "n_per_suite": a.n, "seed": SEED, "bodi_options": bopts,
           "checkpoint": "convaiinnovations/laya (English, ModernBERT-large) - laya_torch loads it from HF; bodi runs models/laya-v2/model.onnx exported from it (identical logits to the first export, plus a pooled output)",
           "power": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0:1],
           "date": time.strftime("%Y-%m-%d %H:%M")}
    print(json.dumps(env), flush=True)
    res = {"env": env, "suites": {}}
    if os.path.exists(a.out) and only:
        res["suites"] = json.load(open(a.out)).get("suites", {})
        res["env_previous"] = json.load(open(a.out)).get("env")

    agent = Agent("convaiinnovations/laya", device="cpu")
    b = Bodi()
    b.load_laya(MODEL_DIR, intra_threads=a.threads)

    for name, S in build_suites(a.n, only).items():
        r = {"n": len(S["states"]), "gold": S["gold"]}
        rows_l, el, dropped = laya_rows(agent, S)
        r["laya_torch"] = summarize(rows_l, el, {"dropped": dropped, "rows_per_decision": 1.0})
        r["laya_torch_pred"] = preds(rows_l)
        print(name, "laya_torch", {k: v for k, v in r["laya_torch"].items()}, flush=True)

        rows_b, el, nrows, gated = bodi_rows(b, S, bopts)
        r["bodi"] = summarize(rows_b, el, {"rows_per_decision": round(nrows / len(rows_b), 3), "script_gated": sum(gated)})
        if any(gated):
            keep = [i for i, x in enumerate(gated) if not x]
            r["non_gated_subset"] = {"n": len(keep),
                                     "laya_torch_accuracy": round(float(np.mean([int(np.argmax(rows_l[i][1])) == rows_l[i][0] for i in keep])), 4),
                                     "bodi_accuracy": round(float(np.mean([int(np.argmax(rows_b[i][1])) == rows_b[i][0] for i in keep])), 4)}
        r["bodi_pred"] = preds(rows_b)
        cl = [p == g for p, g in zip(r["laya_torch_pred"], S["gold"])]
        cb = [p == g for p, g in zip(r["bodi_pred"], S["gold"])]
        r["mcnemar_bodi_vs_laya_torch"] = mcnemar(cb, cl)
        print(name, "bodi", r["bodi"], r["mcnemar_bodi_vs_laya_torch"], flush=True)

        if name == "banking77":
            if "head512" not in skip:
                saved = agent.cfg.get("head_max_len", 192)
                agent.cfg["head_max_len"] = 512
                rows, el, dropped = laya_rows(agent, S)
                agent.cfg["head_max_len"] = saved
                r["laya_head512"] = summarize(rows, el, {"dropped": dropped})
                r["laya_head512_pred"] = preds(rows)
                print(name, "laya_head512", r["laya_head512"], flush=True)
            from laya.shortlist import embed_fn_from_agent, predict_shortlist
            embedders = {}
            if "shortlist_agent" not in skip:
                embedders["laya_shortlist_agent_k20"] = embed_fn_from_agent(agent)
            if "shortlist_minilm" not in skip:
                from sentence_transformers import SentenceTransformer
                st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
                embedders["laya_shortlist_minilm_k20"] = lambda texts: st.encode(list(texts), batch_size=64, convert_to_numpy=True)
            for key, ef in embedders.items():
                rows = []
                t0 = time.time()
                for st_, g in zip(S["states"], S["gold"]):
                    out = predict_shortlist(agent, st_, {S["qid"]: S["q"]}, ef, k=20)
                    probs = out["answers"][S["qid"]]["probabilities"]
                    rows.append((g, [probs.get(l, 0.0) for l in S["labels"]]))
                r[key] = summarize(rows, time.time() - t0)
                r[key + "_pred"] = preds(rows)
                cs = [p == g for p, g in zip(r[key + "_pred"], S["gold"])]
                r["mcnemar_bodi_vs_" + key] = mcnemar(cb, cs)
                print(name, key, r[key], r["mcnemar_bodi_vs_" + key], flush=True)
        res["suites"][name] = r
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
