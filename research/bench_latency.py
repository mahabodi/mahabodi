"""Latency/throughput: Laya vs MahaBodi on the same machine and the same exported weights.

Single-call latency (batch 1): systems are called round-robin on the same case, rotating the
starting system each case, so thermal/turbo drift hits every system equally. Warm-up calls are
excluded. p50/p95 carry bootstrap 95% CIs (2000 resamples, seed 0).

  laya_torch    laya.Agent.predict (PyTorch, CPU, torch.set_num_threads(T))
  laya_onnx     laya.onnx_agent.ONNXAgent.predict on models/laya-v2/model.onnx (ORT default
                threads = physical cores)
  bodi_predict  MahaBodi .predict (Laya-identical maths), ORT intra_threads=T
  bodi_decide   MahaBodi .decide defaults, cache OFF

Throughput: the same N states in one call - laya.Agent.predict_batch vs MahaBodi decide_batch.

Run only on an otherwise idle machine; the script records load average before and after.

    .venv/bin/python research/bench_latency.py --calls 200 --out research/results/latency.json
"""
import argparse, json, os, subprocess, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from laya import Agent
from laya.onnx_agent import ONNXAgent
from mahabodi import Bodi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "laya-v2")


def ci(xs, q, B=2000):
    rng = np.random.default_rng(0)
    xs = np.asarray(xs)
    v = [np.percentile(rng.choice(xs, len(xs)), q) for _ in range(B)]
    return [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)]


def loadavg():
    return os.getloadavg()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "latency.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0).select(range(a.calls + a.warmup + a.batch))
    q = {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                   "criteria": {"world": "world news and international politics", "sports": "sports",
                                "business": "business and economy", "sci_tech": "science and technology"}}}
    states = [{"article": r["text"]} for r in d]

    agent = Agent("convaiinnovations/laya", device="cpu")
    ckpt = snapshot_download("convaiinnovations/laya", allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"])
    oa = ONNXAgent(ckpt, onnx_path=os.path.join(MODEL, "model.onnx"))
    b = Bodi()
    b.load_laya(MODEL, intra_threads=a.threads)
    systems = {
        "laya_torch": lambda s: agent.predict(s, q),
        "laya_onnx": lambda s: oa.predict(s, q),
        "bodi_predict": lambda s: b.predict(s, q),
        "bodi_decide": lambda s: b.decide(s, q, cache=False),
    }
    names = list(systems)
    res = {"env": {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                   "threads": a.threads, "power": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[:2],
                   "loadavg_before": loadavg(), "calls": a.calls, "warmup": a.warmup, "date": time.strftime("%Y-%m-%d %H:%M")}}
    for s in states[:a.warmup]:
        for n in names:
            systems[n](s)
    times = {n: [] for n in names}
    for i, s in enumerate(states[a.warmup:a.warmup + a.calls]):
        order = names[i % len(names):] + names[:i % len(names)]
        for n in order:
            t = time.perf_counter(); systems[n](s); times[n].append((time.perf_counter() - t) * 1000)
    res["single"] = {n: {"p50_ms": round(float(np.percentile(t, 50)), 1), "p50_ci95": ci(t, 50),
                         "p95_ms": round(float(np.percentile(t, 95)), 1), "p95_ci95": ci(t, 95),
                         "mean_ms": round(float(np.mean(t)), 1), "n": len(t), "raw_ms": [round(x, 2) for x in t]} for n, t in times.items()}
    for n in names:
        print(n, {k: v for k, v in res["single"][n].items() if k != "raw_ms"}, flush=True)

    batch = states[a.warmup + a.calls:]
    thr = {}
    for rep in range(3):
        for n, fn in (("laya_torch_predict_batch", lambda: agent.predict_batch(batch, q)),
                      ("bodi_decide_batch", lambda: b.decide_batch(batch, q, cache=False))):
            t = time.perf_counter(); fn(); el = time.perf_counter() - t
            thr.setdefault(n, []).append(len(batch) / el)
    res["throughput_decisions_per_s"] = {n: {"runs": [round(x, 2) for x in v], "median": round(float(np.median(v)), 2)} for n, v in thr.items()}
    res["env"]["loadavg_after"] = loadavg()
    print(res["throughput_decisions_per_s"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
