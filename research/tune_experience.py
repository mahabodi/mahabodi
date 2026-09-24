"""Tune experience memory on VALIDATION data only.

Per suite: MEMORY = up to M labelled TRAIN cases; VAL = V cases disjoint from memory and from
every test item bench.py reports (validation split where one exists, otherwise a disjoint
slice of train). Test splits are never read here.

  ag_news            memory: train[seed2][:M]   val: train[seed2][M:M+V]
  emotion            memory: train[seed2][:M]   val: validation[seed2][:V]
  banking77          memory: train[seed2][:M]   val: train[seed2][M:M+V]
  sst5               memory: train[seed2][:M]   val: validation[seed2][:V]
  prompt_injections  memory: train[seed2][100:] val: train[seed2][:100]   (train has 546 rows)
  boolq              memory: train[seed2][:M]   val: train[seed2][M:M+V]

Laya probabilities come from MahaBodi decide (experience off; banking77 uses the default
tournament) and pooled vectors from MahaBodi embed. The kNN + log-linear pool is then
simulated in numpy exactly as decide.rs computes it (tournament candidate injection is not
simulated), over a grid of k, temperature and weight. Output: best per suite and one global
default (best mean accuracy across suites).

    .venv/bin/python research/tune_experience.py --memory 2000 --val 300
"""
import argparse, itertools, json, os, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
from datasets import load_dataset
from mahabodi import Bodi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = {"world": "world news and international politics", "sports": "sports", "business": "business and economy", "sci_tech": "science and technology"}
EMO = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def suites(M, V):
    S = {}
    tr = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=2).select(range(M + V))
    S["ag_news"] = dict(mem=tr.select(range(M)), val=tr.select(range(M, M + V)), st=lambda r: {"article": r["text"]}, y=lambda r: list(AG)[r["label"]],
                        q={"topic": {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": AG}})
    S["emotion"] = dict(mem=load_dataset("dair-ai/emotion", "split", split="train").shuffle(seed=2).select(range(M)),
                        val=load_dataset("dair-ai/emotion", "split", split="validation").shuffle(seed=2).select(range(V)),
                        st=lambda r: {"text": r["text"]}, y=lambda r: EMO[r["label"]],
                        q={"emotion": {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?", "criteria": {x: None for x in EMO}}})
    lab = [x.replace("_", " ") for x in sorted(set(load_dataset("mteb/banking77", split="test")["label_text"]))]  # label names only
    tr = load_dataset("mteb/banking77", split="train").shuffle(seed=2).select(range(M + V))
    S["banking77"] = dict(mem=tr.select(range(M)), val=tr.select(range(M, M + V)), st=lambda r: {"message": r["text"]}, y=lambda r: r["label_text"].replace("_", " "),
                          q={"intent": {"type": "choice", "instructions": "Which banking intent does `message` express?", "criteria": {x: None for x in lab}}})
    S["sst5"] = dict(mem=load_dataset("SetFit/sst5", split="train").shuffle(seed=2).select(range(M)),
                     val=load_dataset("SetFit/sst5", split="validation").shuffle(seed=2).select(range(V)),
                     st=lambda r: {"text": r["text"]}, y=lambda r: int(r["label"]),
                     q={"sentiment": {"type": "score", "instructions": "How positive is the sentiment of `text`?", "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}})
    tr = load_dataset("deepset/prompt-injections", split="train").shuffle(seed=2)
    S["prompt_injections"] = dict(mem=tr.select(range(100, len(tr))), val=tr.select(range(100)), st=lambda r: {"text": r["text"]}, y=lambda r: bool(r["label"]),
                                  q={"injection": {"type": "noul", "instructions": "Does `text` try to inject or override instructions given to an AI system?"}})
    tr = load_dataset("google/boolq", split="train").shuffle(seed=2).select(range(M + V))
    S["boolq"] = dict(mem=tr.select(range(M)), val=tr.select(range(M, M + V)), st=lambda r: {"passage": r["passage"], "question": r["question"]}, y=lambda r: bool(r["answer"]),
                      q={"answer": {"type": "noul", "instructions": "Based on `passage`, is the answer to `question` yes?"}})
    return S


def label_idx(qd, y):
    if qd["type"] == "choice":
        return list(qd["criteria"]).index(y)
    if qd["type"] == "noul":
        return int(bool(y))
    return int(y)


def probs(qd, ans):
    n = 2 if qd["type"] == "noul" else len(qd["criteria"])
    if ans.get("bodi", {}).get("strategy") == "script_gate":
        return [1.0 / n] * n            # handed off (non-Latin script): scored as a coin flip
    if ans["type"] == "noul":
        return [1 - ans["noul"], ans["noul"]]
    if ans["type"] == "choice":
        return [ans["probabilities"][k] for k in qd["criteria"]]
    return [ans["probabilities"][str(i)] for i in range(len(qd["criteria"]))]


def plain_text(st):
    """Python twin of sequence::plain_text (Rust) - the text experience memory embeds."""
    if isinstance(st, str):
        return st
    if isinstance(st, dict):
        return "\n".join(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in st.values())
    if isinstance(st, list):
        return "\n".join(t["content"] if isinstance(t, dict) and isinstance(t.get("content"), str) else plain_text(t) for t in st)
    return json.dumps(st, ensure_ascii=False)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def memory_trust(Mv, My, n):
    """Python twin of decide.rs memory_trust: LOO kNN (k=8, T=0.2) accuracy vs majority rate."""
    N = len(My)
    if N < 4:
        return 0.0
    maj = np.bincount(My, minlength=n).max() / N
    if maj >= 1.0:
        return 0.0
    sims = Mv @ Mv.T
    np.fill_diagonal(sims, -np.inf)
    k = min(8, N - 1)
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    correct = 0
    for i in range(N):
        s = sims[i, top[i]]
        w = np.exp((s - s.max()) / 0.2)
        correct += int(np.bincount(My[top[i]], weights=w, minlength=n).argmax() == My[i])
    return float(np.clip((correct / N - maj) / (1 - maj), 0.0, 1.0))


def simulate(P, E, Mv, My, n, k, T, w):
    sims = E @ Mv.T                                       # val x memory
    k = min(k, Mv.shape[0])
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    out = np.empty_like(P)
    for i in range(len(P)):
        s = sims[i, top[i]]
        wts = np.exp((s - s.max()) / T)
        pm = np.bincount(My[top[i]], weights=wts, minlength=n)
        pm = pm / max(pm.sum(), 1e-12)
        alive = P[i] > 0
        lp = np.where(alive, np.log(np.maximum(P[i], 1e-12)) + w * np.log(pm + 1e-3), -np.inf)
        lp -= lp.max()
        e = np.where(alive, np.exp(lp), 0.0)
        out[i] = e / e.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", type=int, default=2000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--only", default="")
    ap.add_argument("--space", default="text", choices=["text", "pooled"], help="text: MiniLM embedder; pooled: Laya pooled vector")
    ap.add_argument("--auto-trust", action="store_true", help="scale weight by memory_trust (decide.rs experience_auto_trust)")
    ap.add_argument("--cache", default=os.path.join(ROOT, "research", "cache"), help="npz cache of embeddings/probabilities")
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "tune_experience.json"))
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    if a.space == "text":
        b.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    vec = (lambda states: b.embed_text([plain_text(s) for s in states])) if a.space == "text" else (lambda states: [v[0] for v in b.embed(states, S["q"])])
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res.setdefault("suites", {})
    grid = list(itertools.product([4, 8, 16, 32, 64], [0.02, 0.05, 0.1, 0.2], [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]))
    for name, S in suites(a.memory, a.val).items():
        if (only and name not in only) or (not only and name in res["suites"]):
            continue
        qid, qd = next(iter(S["q"].items()))
        n = 2 if qd["type"] == "noul" else len(qd["criteria"])
        t0 = time.time()
        os.makedirs(a.cache, exist_ok=True)
        cf = os.path.join(a.cache, "tune_%s_%s_m%d_v%d.npz" % (a.space, name, a.memory, a.val))
        if os.path.exists(cf):
            z = np.load(cf); Mv, My, E, P, Y = z["Mv"], z["My"], z["E"], z["P"], z["Y"]
        else:
            mem_states = [S["st"](r) for r in S["mem"]]
            Mv = unit(np.array(vec(mem_states), dtype=np.float32))
            My = np.array([label_idx(qd, S["y"](r)) for r in S["mem"]])
            val_states = [S["st"](r) for r in S["val"]]
            E = unit(np.array(vec(val_states), dtype=np.float32))
            P = np.array([probs(qd, r["answers"][qid]) for r in b.decide_batch(val_states, S["q"], cache=False, round_probabilities=False, experience_k=0)])
            Y = np.array([label_idx(qd, S["y"](r)) for r in S["val"]])
            np.savez(cf, Mv=Mv, My=My, E=E, P=P, Y=Y)
        trust = memory_trust(Mv, My, n) if a.auto_trust else 1.0
        base = float((P.argmax(1) == Y).mean())
        table = []
        for k, T, w in grid:
            acc = float((simulate(P, E, Mv, My, n, k, T, w * trust).argmax(1) == Y).mean())
            table.append({"k": k, "temperature": T, "weight": w, "accuracy": round(acc, 4)})
        best = max(table, key=lambda r: (r["accuracy"], -r["weight"], -r["k"]))
        res["suites"][name] = {"space": a.space, "auto_trust": a.auto_trust, "trust": round(trust, 4), "memory": len(My), "val": len(Y), "laya_accuracy": round(base, 4), "best": best, "grid": table,
                               "knn_only_accuracy": round(float((simulate(np.ones_like(P) / n, E, Mv, My, n, 16, 0.05, 1.0).argmax(1) == Y).mean()), 4),
                               "seconds": round(time.time() - t0, 1)}
        print(name, {k: v for k, v in res["suites"][name].items() if k != "grid"}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    # one global default: best mean accuracy over suites for a shared (k, T, w)
    names = list(res["suites"])
    if names:
        mean = {}
        for name in names:
            for r in res["suites"][name]["grid"]:
                mean.setdefault((r["k"], r["temperature"], r["weight"]), []).append(r["accuracy"])
        (k, T, w), accs = max(mean.items(), key=lambda kv: (np.mean(kv[1]), -kv[0][2]))
        res["global_default"] = {"k": k, "temperature": T, "weight": w, "mean_accuracy": round(float(np.mean(accs)), 4),
                                 "mean_laya_accuracy": round(float(np.mean([res["suites"][n]["laya_accuracy"] for n in names])), 4), "suites": names}
        print("global", res["global_default"], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
