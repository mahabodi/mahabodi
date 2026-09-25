"""Latency: Laya vs MahaBodi on the SAME machine, CPU only (pre-registered with the reviewer before
measuring). Laya's published 32.8 ms is on a T4 GPU and is NOT compared.

Protocol:
  - CPU only for both. Same thread count T (default 8): torch.set_num_threads(T) for Laya, ONNX Runtime
    intra_threads=T for MahaBodi.
  - Batch size 1, one question per call, fixed inputs (seeded AG News test texts).
  - 50 warm-up calls per system (excluded), then 500 timed decisions per system, round-robin on the same
    case with the starting system rotated each case, so drift hits all systems equally.
  - Report p50 / p95 / mean in ms, with bootstrap 95% CIs (2000 resamples, seed 0) for p50 and p95.
    Verdict per pair: faster only when the p50 CIs do not overlap.
  - Rows:
      ag_news (4 options): laya_torch  vs  bodi_decide (defaults, cache off); also bodi_predict
                           (Laya-identical maths) and laya_onnx (Laya's own ONNXAgent; its ONNX Runtime
                           uses its default thread count, which the agent does not expose, so it is a
                           REFERENCE row, not thread-matched)
      banking77 (77 options), separate row: laya_torch (1 pass) vs bodi_decide (tournament, ~4 passes)
  - Idle box: load average recorded before and after; machine provenance recorded.

    ORT_DYLIB_PATH=... .venv/bin/python research/bench_latency.py --out research/results/latency_<host>.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites  # noqa: E402
from provenance import provenance  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from laya import Agent  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "laya-v2")


def ci(xs, q, B=2000):
    rng = np.random.default_rng(0)
    xs = np.asarray(xs)
    v = [np.percentile(rng.choice(xs, len(xs)), q) for _ in range(B)]
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]


def measure(systems, states, warmup, calls):
    names = list(systems)
    for s in states[:warmup]:
        for n in names:
            systems[n](s)
    times = {n: [] for n in names}
    for i, s in enumerate(states[warmup:warmup + calls]):
        order = names[i % len(names):] + names[:i % len(names)]
        for n in order:
            t = time.perf_counter(); systems[n](s); times[n].append((time.perf_counter() - t) * 1000)
    return {n: {"p50_ms": round(float(np.percentile(t, 50)), 2), "p50_ci95": ci(t, 50),
                "p95_ms": round(float(np.percentile(t, 95)), 2), "p95_ci95": ci(t, 95),
                "mean_ms": round(float(np.mean(t)), 2), "n": len(t), "raw_ms": [round(x, 3) for x in t]} for n, t in times.items()}


def faster(a, b):
    if a["p50_ci95"][1] < b["p50_ci95"][0]:
        return "faster"
    if a["p50_ci95"][0] > b["p50_ci95"][1]:
        return "slower"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = {"protocol": __doc__.split("\n\n")[1], "threads": a.threads, "calls": a.calls, "warmup": a.warmup,
           "loadavg_before": os.getloadavg(), "provenance": provenance(),
           "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_cuda_available": torch.cuda.is_available()}}
    n = a.warmup + a.calls
    S = build_suites(max(n, 500), {"ag_news", "banking77"})
    agent = Agent("convaiinnovations/laya", device="cpu")
    ckpt = snapshot_download("convaiinnovations/laya", allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"])
    oa = ONNXAgent(ckpt, onnx_path=os.path.join(MODEL, "model.onnx"))
    b = Bodi(); b.load_laya(MODEL, intra_threads=a.threads)
    for suite, extra in (("ag_news", True), ("banking77", False)):
        s = S[suite]; q = {s["qid"]: s["q"]}
        states = (s["states"] * (n // len(s["states"]) + 1))[:n]
        systems = {"laya_torch": lambda st: agent.predict(st, q), "bodi_decide": lambda st: b.decide(st, q, cache=False)}
        if extra:
            systems["bodi_predict"] = lambda st: b.predict(st, q)
            systems["laya_onnx_reference"] = lambda st: oa.predict(st, q)
        r = measure(systems, states, a.warmup, a.calls)
        r["verdict_bodi_decide_vs_laya_torch"] = faster(r["bodi_decide"], r["laya_torch"])
        res[suite] = r
        print(suite, {k: ({kk: vv for kk, vv in v.items() if kk != "raw_ms"} if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
    res["loadavg_after"] = os.getloadavg()
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
