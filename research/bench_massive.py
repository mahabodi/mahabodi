"""MASSIVE intent, 51 languages, 20 options - Laya's multilingual benchmark (published: laya-multilingual
has 45/51 languages above 3x random = accuracy > 0.15; macro accuracy 0.3661).

Construction is copied from Laya's research/scripts/bench_local.py `build_massive` (first `per_lang`
test rows per language, gold + 19 distractors drawn with random.Random(13), shuffled), because that
construction defines the published number. Both systems use the laya-multilingual checkpoint.

  laya_torch   Laya PyTorch multilingual checkpoint, Laya's own score_cases
  bodi         MahaBodi defaults (cache off) on models/laya-multilingual. 20 options > the default
               max_options_per_pass (16), so MahaBodi's tournament runs; its settings were fixed on
               English banking77 validation and are NOT tuned on MASSIVE.

Outputs per-language accuracy for both, languages above 0.15, macro accuracy, and an exact McNemar
over all items plus per language.

    .venv/bin/python research/bench_massive.py --per-lang 100 --out research/results/bench_massive.json
"""
import argparse, json, os, random, re, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import LAYA_COMMIT, LAYA_SRC, mcnemar, wilson  # noqa: E402
sys.path.insert(0, os.path.join(LAYA_SRC, "research", "scripts"))
from bench_local import metrics, score_cases, softmax_t, temp_for  # noqa: E402
from laya import Agent  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, N_OPTS = 13, 20


def languages():
    from huggingface_hub import HfApi
    files = [s.rfilename for s in (HfApi().dataset_info("mteb/amazon_massive_intent").siblings or [])]
    return sorted({m.group(1) for f in files for m in [re.match(r"test/([A-Za-z\-]+)\.json", f)] if m})


def build(lg, per_lang):
    from datasets import load_dataset
    d = load_dataset("mteb/amazon_massive_intent", lg, split="test")
    labels = sorted(set(d["label_text"]))
    rng = random.Random(SEED)
    cases, gold = [], []
    for r in list(d)[:per_lang]:
        pool = [x for x in labels if x != r["label_text"]]
        keys = [r["label_text"]] + rng.sample(pool, min(N_OPTS - 1, len(pool)))
        rng.shuffle(keys)
        cases.append(({"utterance": r["text"]}, {"intent": {"type": "choice", "instructions": "What is the user asking for in `utterance`?",
                                                             "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys}}}))
        gold.append(keys.index(r["label_text"]))
    return cases, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "bench_massive.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"per_language": {}}
    res.update({"laya_commit": LAYA_COMMIT, "baseline": "laya-multilingual on all languages (not Router-routed)", "per_lang": a.per_lang, "n_options": N_OPTS, "seed": SEED, "threshold_3x_random": 3.0 / N_OPTS})
    agent = Agent("convaiinnovations/laya", subfolder="multilingual", device="cpu")
    b = Bodi(); b.load_laya(os.path.join(ROOT, "models", "laya-multilingual"), intra_threads=a.threads)
    for lg in languages():
        if lg in res["per_language"]:
            continue
        cases, gold = build(lg, a.per_lang)
        t0 = time.time()
        lgs, idx, _, dropped = score_cases(agent, cases)
        tl = time.time() - t0
        lp = [None if z is None else int(np.argmax(softmax_t(z, temp_for(agent, qt, k)))) for (ci, _, qt, k), z in zip(idx, lgs)]
        t0 = time.time()
        bp = []
        for st, q in cases:  # questions differ per case (distractors), so one decide per case
            ans = b.decide(st, q, cache=False)["answers"]["intent"]
            bp.append(None if ans["choice"] is None else list(q["intent"]["criteria"]).index(ans["choice"]))
        tb = time.time() - t0
        cl = [p == g for p, g in zip(lp, gold)]; cb = [p == g for p, g in zip(bp, gold)]
        res["per_language"][lg] = {"laya_accuracy": round(float(np.mean(cl)), 4), "bodi_accuracy": round(float(np.mean(cb)), 4),
                                   "laya_ci95": wilson(sum(cl), len(cl)), "bodi_ci95": wilson(sum(cb), len(cb)),
                                   "mcnemar_bodi_vs_laya": mcnemar(cb, cl), "laya_seconds": round(tl, 1), "bodi_seconds": round(tb, 1),
                                   "dropped": dropped, "gold": gold, "laya_pred": lp, "bodi_pred": bp}
        print(lg, {k: v for k, v in res["per_language"][lg].items() if k not in ("gold", "laya_pred", "bodi_pred")}, flush=True)
        json.dump(res, open(a.out, "w"))
    P = res["per_language"]
    for sysname in ("laya", "bodi"):
        accs = [v[sysname + "_accuracy"] for v in P.values()]
        res[sysname] = {"macro_accuracy": round(float(np.mean(accs)), 4), "languages_above_3x_random": int(sum(x > 3.0 / N_OPTS for x in accs)), "n_languages": len(accs)}
    allc = lambda s: [p == g for v in P.values() for p, g in zip(v[s + "_pred"], v["gold"])]
    res["mcnemar_bodi_vs_laya_all_items"] = mcnemar(allc("bodi"), allc("laya"))
    print("laya", res["laya"], "bodi", res["bodi"], res["mcnemar_bodi_vs_laya_all_items"], flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
