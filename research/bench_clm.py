"""CLM-8B on the same seeded test items as Laya and MahaBodi (research/PREREG_CLM.md).

CLM runs via its own `clm.Engine` (repo bb42c6c, reference head CLM-v0.1-8B) against research/clm_embed_server.py
(Qwen3-8B on CPU, vLLM pooling recipe). Questions are research/bench.py's wire-format questions; `criteria: None`
descriptions become "" (CLM then embeds the option key, per its API docs). The predicted index is the argmax of the
returned probabilities (the same rule used for Laya). A noul is "true" when p >= 0.5.

    --parity: reproduce CLM's README example outputs (the end-to-end encoder parity check, run before any scoring)
    default: score the suites and write research/results/bench_clm.json (per item: label, probabilities, latency)

    CLM_HEAD=<.pt> .venv/bin/python research/bench_clm.py --parity
    CLM_HEAD=<.pt> .venv/bin/python research/bench_clm.py --suites ag_news,emotion,banking77,sst5,prompt_injections,boolq
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
README_EXAMPLE = {
    "state": "Customer: my invoice was charged twice and nobody answers the phone!",
    "questions": {"urgency": {"type": "noul", "instructions": "Is this urgent?"},
                  "department": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
                  "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                                  "criteria": ["Calm", "Frustrated", "Very angry"]}},
    "expected": {"urgency.noul": 0.41022, "department.billing": 0.93878, "frustration.score": 1.98386},
}


def engine():
    from clm import Engine
    return Engine(checkpoint=os.environ["CLM_HEAD"], device="cpu",
                  emb_url=os.environ.get("CLM_EMB_URL", "http://127.0.0.1:8090/v1/embeddings"), action_cache=0)


def as_dict(a):
    return a if isinstance(a, dict) else (a.__dict__ if hasattr(a, "__dict__") else dict(a))


def parity(e):
    r = e.answer(README_EXAMPLE["state"], README_EXAMPLE["questions"])
    A = {k: as_dict(v) for k, v in r["answers"].items()}
    got = {"urgency.noul": A["urgency"]["noul"], "department.billing": A["department"]["probabilities"]["billing"],
           "frustration.score": A["frustration"]["score"]}
    diff = {k: round(abs(got[k] - v), 5) for k, v in README_EXAMPLE["expected"].items()}
    ok = all(d <= 0.005 for d in diff.values())
    return {"expected": README_EXAMPLE["expected"], "got": {k: round(v, 5) for k, v in got.items()}, "abs_diff": diff,
            "tolerance": 0.005, "pass": ok, "raw": r.get("answers") and {k: as_dict(v) for k, v in r["answers"].items()}}


def clm_question(q):
    q = json.loads(json.dumps(q))
    if isinstance(q.get("criteria"), dict):
        q["criteria"] = {k: (v or "") for k, v in q["criteria"].items()}
    return q


def pred_index(ans, q, labels):
    a = as_dict(ans)
    if q["type"] == "noul":
        return int(a["noul"] >= 0.5), {"true": a["noul"]}
    probs = a["probabilities"]
    if q["type"] == "score":
        vec = [probs[str(i)] if str(i) in probs else probs[i] if i in probs else probs[q["criteria"][i]] for i in range(len(q["criteria"]))]
        return int(np.argmax(vec)), probs
    return int(np.argmax([probs[l] for l in labels])), probs


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--suites", default="ag_news,emotion,banking77,sst5,prompt_injections,boolq")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(R, "bench_clm.json"))
    a = ap.parse_args()
    t0 = time.time(); e = engine(); head_load_s = time.time() - t0
    if a.parity:
        p = parity(e); p2 = parity(e)
        p["deterministic_repeat"] = p["got"] == p2["got"]
        json.dump({"parity": p, "provenance": provenance(), "head": os.environ["CLM_HEAD"], "head_sha256": sha(os.environ["CLM_HEAD"])},
                  open(os.path.join(R, "clm_parity.json"), "w"), indent=1, default=str)
        print(json.dumps({k: p[k] for k in ("expected", "got", "abs_diff", "pass", "deterministic_repeat")}))
        return
    from bench import build_suites, wilson, mcnemar
    base = json.load(open(os.path.join(R, "bench.json")))["suites"]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"suites": {}}
    res.update({"protocol": "research/PREREG_CLM.md", "provenance": provenance(), "head": os.environ["CLM_HEAD"],
                "head_sha256": sha(os.environ["CLM_HEAD"]), "head_load_s": round(head_load_s, 1)})
    for name in a.suites.split(","):
        if name in res["suites"]:
            continue
        S = build_suites(a.n, {name})[name]
        q = clm_question(S["q"]); assert S["gold"] == base[name]["gold"], "items differ from bench.json"
        preds, probs, lat = [], [], []
        for st in S["states"]:
            t = time.perf_counter(); r = e.answer(st, {S["qid"]: q}); lat.append((time.perf_counter() - t) * 1000)
            i, pr = pred_index(r["answers"][S["qid"]], S["q"], S["labels"]); preds.append(i); probs.append(pr)
        # determinism: the first 50 items again
        rep = [pred_index(e.answer(st, {S["qid"]: q})["answers"][S["qid"]], S["q"], S["labels"])[0] for st in S["states"][:50]]
        g = S["gold"]; c = [p_ == y for p_, y in zip(preds, g)]
        lay = [p_ == y for p_, y in zip(base[name]["laya_torch_pred"], g)]; bod = [p_ == y for p_, y in zip(base[name]["bodi_pred"], g)]
        lat_s = sorted(lat)
        res["suites"][name] = {"n": len(g), "accuracy": round(sum(c) / len(g), 4), "ci95": wilson(sum(c), len(g)),
                               "laya_accuracy": round(sum(lay) / len(g), 4), "mahabodi_accuracy": round(sum(bod) / len(g), 4),
                               "mcnemar_clm_vs_laya": mcnemar(c, lay), "mcnemar_clm_vs_mahabodi": mcnemar(c, bod),
                               "deterministic_first50": rep == preds[:50],
                               "latency_ms_p50_accuracy_run": lat_s[len(lat_s) // 2], "latency_ms_p95_accuracy_run": lat_s[int(.95 * len(lat_s))],
                               "latency_note": "accuracy-run timings (options cached after the first item, the state always new); the fresh same-machine latency comparison is a separate run",
                               "pred": preds, "probabilities": probs}
        print(name, {k: v for k, v in res["suites"][name].items() if k not in ("pred", "probabilities")}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
