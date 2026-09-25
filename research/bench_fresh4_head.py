"""A trained Laya head as a MahaBodi component (pre-registered with the reviewer before running).

Candidates per suite, all on the same 2,000 labelled examples:
  A  current MahaBodi: base Laya (models/laya-v2) + memory, learn(calibrate=200), decide() defaults
  B  the fine-tuned head alone: models/laya-head-<suite> (finetune_laya_head.py --save-head, exported by
     export_onnx.py --head-state), Laya-identical predict()
  C  the fine-tuned head + MahaBodi memory: models/laya-head-<suite>, learn(calibrate=200), decide() defaults

Selection (fixed before running): the highest accuracy on a NEW selection set, disjoint from the memory,
the fine-tune's train/validation data and every test/fresh sample (asserted by index and by text hash):
train.shuffle(2) rows [2000:2300] where the validation split is separate (emotion, sst5), [2300:2600] where
validation came from train (ag_news, banking77, boolq). If the best candidate beats A by less than 1 point,
A is kept (no training step). All three selection accuracies are recorded.

Test ONCE on a 4th fresh sample: seeded-shuffle positions 1500..1999 (asserted: >= 2000 rows, disjoint from
0..1499). Duplicates are DROPPED (fixed before running): texts are normalised (strip, lowercase, collapse
whitespace); every fresh-4 or selection item whose normalised text occurs in the memory [:2000] (also the
fine-tune's training data), the fine-tune's validation data, or test/fresh positions 0..1499 is removed
(selection items also if they occur in fresh-4). Removed counts are recorded; claims use the remaining items
(n reported); accuracy on the full 500 including duplicates is reported as a secondary, non-claim number. Reported on it: A, B, C, the selected candidate, Laya zero-shot (MahaBodi's Laya-identical predict,
verified 0 discordance with Laya's PyTorch model), and plain kNN (own-tuned k / T). Exact McNemar for selected
vs B (the fine-tuned head), selected vs Laya, selected vs kNN, and C vs B (does memory add on top of a trained
head?).

Two exported heads (fixed before running): where a suite's head re-run differs significantly (p < 0.05) from
its run of record (banking77: 0.558 vs 0.598 on fresh3, 37/57, p 0.049), both heads are exported
(models/laya-head-<suite> = the re-run, models/laya-head-<suite>-record = the run of record, reproduced) and
B is the one with the higher SELECTION-set accuracy (no fresh/test data used); ties go to the smaller lr, then
fewer epochs. C uses the same head as B. Both heads' selection accuracies are recorded.

Circularity, stated: B IS the fine-tuned head, so "selected >= fine-tuned head" is near-true by construction
whenever B or C is selected. The informative results are (i) whether selection picked well (selected vs
max(A, B, C) on fresh) and (ii) C vs B. Where B or C is selected, MahaBodi includes a training step for that
suite (cost recorded from the fine-tune file); the "no training step" property applies only to A.
prompt_injections has no fresh items and no room for a selection set: A, B and C are reported on its 116 test
items only (same duplicate rule against its memory and validation data), with C vs B and B vs A; no candidate
is selected and nothing is claimed beyond those test-only numbers.

    .venv/bin/python research/bench_fresh4_head.py --out research/results/bench_fresh4_head.json
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites, mcnemar, wilson  # noqa: E402
from provenance import provenance  # noqa: E402
from tune_experience import suites as train_suites, label_idx, plain_text  # noqa: E402
from datasets import load_dataset  # noqa: E402
from mahabodi import Bodi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")
M = os.path.join(ROOT, "models")
SUITES = ["emotion", "banking77", "sst5", "ag_news", "boolq"]
TRAIN = {"ag_news": ("fancyzhx/ag_news", None), "banking77": ("mteb/banking77", None), "boolq": ("google/boolq", None),
         "emotion": ("dair-ai/emotion", "split"), "sst5": ("SetFit/sst5", None)}
VAL_FROM_TRAIN = {"ag_news", "banking77", "boolq"}


def norm(st):
    return " ".join(plain_text(st).lower().split())


def preds(b, states, q, labels, mode):
    qid = next(iter(q))
    out = b.decide_batch(states, q, cache=False, round_probabilities=False) if mode == "decide" else [b.predict(s, q) for s in states]
    res = []
    for o in out:
        a = o["answers"][qid]
        if a.get("bodi", {}).get("strategy") == "script_gate":
            res.append(None); continue
        p = [1 - a["noul"], a["noul"]] if a["type"] == "noul" else [a["probabilities"][l] for l in labels]
        res.append(int(np.argmax(p)))
    return res


def acc(p, g):
    return float(np.mean([x == y for x, y in zip(p, g)]))


def pi_test_only(T, S):
    """prompt_injections: no fresh items and no selection set; A, B, C on its test items (duplicates vs memory/val dropped)."""
    qid, qd = next(iter(T["q"].items()))
    used = {norm(T["st"](x)) for x in T["mem"]} | {norm(T["st"](x)) for x in T["val"]}
    keep = [norm(s_) not in used for s_ in S["states"]]
    mem_states = [T["st"](x) for x in T["mem"]]; mem_labels = [{qid: T["y"](x)} for x in T["mem"]]
    P = {}
    for cand, model_dir, mode in (("A", os.path.join(M, "laya-v2"), "decide"), ("B", os.path.join(M, "laya-head-prompt_injections"), "predict"),
                                  ("C", os.path.join(M, "laya-head-prompt_injections"), "decide")):
        b = Bodi(); b.load_laya(model_dir, intra_threads=8); b.load_embedder(os.path.join(M, "minilm"), intra_threads=8)
        if mode == "decide":
            b.learn(mem_states, T["q"], mem_labels, calibrate=200)
        P[cand] = [p for p, k in zip(preds(b, S["states"], T["q"], S["labels"], mode), keep) if k]
    g = [x for x, k in zip(S["gold"], keep) if k]
    c_ = lambda p: [x == y for x, y in zip(p, g)]
    return {"test_only": True, "caveat": "no fresh sample and no selection set; MahaBodi's earlier per-task settings were tuned on these items",
            "removed_duplicates": keep.count(False), "n": len(g), "accuracy": {c: round(acc(P[c], g), 4) for c in "ABC"},
            "mcnemar_C_vs_B": mcnemar(c_(P["C"]), c_(P["B"])), "mcnemar_B_vs_A": mcnemar(c_(P["B"]), c_(P["A"])),
            "pred": P, "gold": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "bench_fresh4_head.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x] or SUITES
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"suites": {}}
    res.update({"protocol": __doc__.split("\n\n")[1:5], "provenance": provenance()})
    tr = train_suites(2000, 300)
    S_all = build_suites(2000, set(only) - {"prompt_injections"})
    kown = json.load(open(os.path.join(R, "tune_knn_only.json")))
    ft_files = [json.load(open(os.path.join(R, f))) for f in ("laya_head_finetuned.json", "laya_head_finetuned_ubuntu.json",
                                                               "laya_head_finetuned_ubuntu_v4.json") if os.path.exists(os.path.join(R, f))]
    if not a.only:
        only = only + ["prompt_injections"]
    for name in only:
        if name in res["suites"]:
            continue
        if name == "prompt_injections":
            res["suites"][name] = pi_test_only(tr[name], build_suites(200, {name})[name])
            json.dump(res, open(a.out, "w"), indent=1)
            continue
        T, S = tr[name], S_all[name]
        qid, qd = next(iter(T["q"].items()))
        assert len(S["gold"]) >= 2000, "%s has only %d items" % (name, len(S["gold"]))
        fresh = dict(S, states=S["states"][1500:2000], gold=S["gold"][1500:2000])
        # selection set, disjoint by index (train-derived) and by text hash (all)
        ds, cfg = TRAIN[name]
        full = load_dataset(ds, cfg, split="train").shuffle(seed=2)
        lo = 2300 if name in VAL_FROM_TRAIN else 2000
        sel_rows = full.select(range(lo, lo + 300))
        sel_states = [T["st"](r) for r in sel_rows]; sel_gold = [label_idx(qd, T["y"](r)) for r in sel_rows]
        assert lo >= 2000 + (300 if name in VAL_FROM_TRAIN else 0), "selection rows overlap memory/val by index"
        used = {norm(T["st"](x)) for x in T["mem"]} | {norm(T["st"](x)) for x in T["val"]} | {norm(s_) for s_ in S["states"][:1500]}
        keep_f = [norm(s_) not in used for s_ in fresh["states"]]
        f4 = {norm(s_) for s_ in fresh["states"]}
        keep_s = [norm(s_) not in used and norm(s_) not in f4 for s_ in sel_states]
        head_dir = os.path.join(M, "laya-head-%s" % name)
        assert os.path.exists(os.path.join(head_dir, "model.onnx")), "missing %s" % head_dir
        mem_states = [T["st"](x) for x in T["mem"]]; mem_labels = [{qid: T["y"](x)} for x in T["mem"]]
        t0 = time.time(); r = {"selection_rows": "train.shuffle(2)[%d:%d]" % (lo, lo + 300), "selection_removed_duplicates": keep_s.count(False),
                               "fresh_removed_duplicates": keep_f.count(False), "fresh_n": sum(keep_f), "selection_n": sum(keep_s)}
        # two exported heads: choose B's head on the SELECTION set only (tie: smaller lr, then fewer epochs)
        rec_dir = head_dir + "-record"
        if os.path.exists(os.path.join(rec_dir, "model.onnx")):
            meta = lambda d: json.load(open(os.path.join(d, "head_meta.json")))
            hs = {}
            for d in (head_dir, rec_dir):
                b = Bodi(); b.load_laya(d, intra_threads=8)
                ps = preds(b, sel_states, T["q"], S["labels"], "predict"); del b
                hs[d] = {"selection_accuracy": round(acc([x for x, k in zip(ps, keep_s) if k], [x for x, k in zip(sel_gold, keep_s) if k]), 4), **meta(d)}
            head_dir = max(hs, key=lambda d: (hs[d]["selection_accuracy"], -hs[d]["lr"], -hs[d]["epoch"]))
            r["two_heads"] = {os.path.basename(d): v for d, v in hs.items()}
            r["head_used"] = os.path.basename(head_dir)
        P = {}
        for cand, model_dir, mode in (("A", os.path.join(M, "laya-v2"), "decide"), ("B", head_dir, "predict"), ("C", head_dir, "decide")):
            b = Bodi(); b.load_laya(model_dir, intra_threads=8); b.load_embedder(os.path.join(M, "minilm"), intra_threads=8)
            if mode == "decide":
                b.learn(mem_states, T["q"], mem_labels, calibrate=200)
            P[cand] = {"selection": preds(b, sel_states, T["q"], S["labels"], mode), "fresh": preds(b, fresh["states"], T["q"], S["labels"], mode)}
            if cand == "A":
                P["laya"] = {"fresh": preds(b, fresh["states"], T["q"], S["labels"], "predict")}
            del b
        K = lambda xs, keep: [x for x, k in zip(xs, keep) if k]
        sel_acc = {c: round(acc(K(P[c]["selection"], keep_s), K(sel_gold, keep_s)), 4) for c in "ABC"}
        best = max("ABC", key=lambda c: (sel_acc[c], c == "A"))
        chosen = best if sel_acc[best] - sel_acc["A"] >= 0.01 else "A"
        # plain kNN (own-tuned) on fresh
        e = Bodi(); e.load_embedder(os.path.join(M, "minilm"), intra_threads=8)
        Mv = np.array(e.embed_text([plain_text(s) for s in mem_states]), dtype=np.float32)
        My = np.array([label_idx(qd, T["y"](x)) for x in T["mem"]])
        E = np.array(e.embed_text([plain_text(s) for s in fresh["states"]]), dtype=np.float32)
        sims = E @ Mv.T; ko = kown[name]; top = np.argsort(-sims, 1)[:, :ko["k"]]; n = len(S["labels"])
        knn = [int(np.bincount(My[t], weights=np.exp((sims[i, t] - sims[i, t].max()) / ko["temperature"]), minlength=n).argmax()) for i, t in enumerate(top)]
        g_all = fresh["gold"]
        full500 = {c: round(acc(P[c]["fresh"], g_all), 4) for c in "ABC"}
        full500.update({"laya": round(acc(P["laya"]["fresh"], g_all), 4), "knn": round(acc(knn, g_all), 4)})
        for c in list(P):
            P[c]["fresh"] = K(P[c]["fresh"], keep_f)
        knn = K(knn, keep_f); g = K(g_all, keep_f); c_ = lambda p: [x == y for x, y in zip(p, g)]
        fr = {c: round(acc(P[c]["fresh"], g), 4) for c in "ABC"}
        fr.update({"laya": round(acc(P["laya"]["fresh"], g), 4), "knn": round(acc(knn, g), 4), "selected": fr[chosen]})
        r["secondary_full_500_including_duplicates"] = full500
        ftcost = next((f["suites"][name].get("train_gpu_seconds") or f["suites"][name].get("encode_seconds") for f in reversed(ft_files) if name in f["suites"]), None)
        r.update({"selection_accuracy": sel_acc, "chosen": chosen, "fresh_accuracy": fr,
                  "fresh_ci95": {k: wilson(sum(c_(P[k]["fresh"])), len(g)) for k in "ABC"},
                  "mcnemar_selected_vs_ft_head": mcnemar(c_(P[chosen]["fresh"]), c_(P["B"]["fresh"])),
                  "mcnemar_selected_vs_laya": mcnemar(c_(P[chosen]["fresh"]), c_(P["laya"]["fresh"])),
                  "mcnemar_selected_vs_knn": mcnemar(c_(P[chosen]["fresh"]), c_(knn)),
                  "mcnemar_C_vs_B": mcnemar(c_(P["C"]["fresh"]), c_(P["B"]["fresh"])),
                  "selection_regret": round(max(fr[c] for c in "ABC") - fr[chosen], 4),
                  "training_step": chosen != "A", "ft_cost_seconds_from_file": ftcost,
                  "pred": {k: P[k]["fresh"] for k in ("A", "B", "C", "laya")}, "knn_pred": knn, "gold": g, "seconds": round(time.time() - t0, 1)})
        res["suites"][name] = r
        print(name, {k: r[k] for k in ("selection_accuracy", "chosen", "fresh_accuracy", "mcnemar_selected_vs_ft_head", "mcnemar_C_vs_B", "selection_regret")}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
