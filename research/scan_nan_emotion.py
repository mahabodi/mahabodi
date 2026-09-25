"""Reviewer check for the full fine-tune's emotion run of record (attempt 2, trained under PyTorch's DEFAULT
SDPA kernels on the Ubuntu GPU, before the NaN-on-padded-rows bug was found). Emotion stays the run of record
only if 0 non-finite logits occur over the batches that run actually used, under that same default kernel:

  - training: batches of 8 in the order rng=np.random.default_rng(seed).permutation gives, for every epoch the
    grid could have run (seed 0: 10 epochs; seeds 1-2: the chosen 5 epochs)
  - validation / test / fresh: batches of 32 sorted by length, as predict() forms them

Forward passes use Laya's base weights: the NaN depends on the kernel and the padding pattern of the batch.

    .venv/bin/python research/scan_nan_emotion.py
"""
import json, os, sys
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_suites  # noqa: E402
from provenance import provenance  # noqa: E402
from finetune_laya_full import items_for  # noqa: E402
from tune_experience import suites as train_suites  # noqa: E402
from laya import Agent  # noqa: E402
from laya.common import collate_items  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")


def count_bad(model, items, batches, dev):
    bad = 0
    with torch.no_grad():
        for sel in batches:
            b = collate_items([[items[i] for i in sel]], 0)
            lg, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev), b["marker_mask"].to(dev), b["qtype"].to(dev))
            bad += int((~torch.isfinite(lg.float()).all(-1)).sum())
    return bad


def main():
    dev = torch.device("cuda")  # default SDPA kernels, as in the emotion run of record
    agent = Agent("convaiinnovations/laya", device="cpu")
    model = agent.model.to(dev).eval()
    T = train_suites(2000, 300)["emotion"]; qid, qd = next(iter(T["q"].items()))
    S = build_suites(1500, {"emotion"})["emotion"]
    Xtr = items_for(agent, [T["st"](r) for r in T["mem"]], qd)
    Xva = items_for(agent, [T["st"](r) for r in T["val"]], qd)
    Xte = items_for(agent, S["states"][:500], S["q"]); Xf = items_for(agent, S["states"][1000:1500], S["q"])
    out = {"provenance": provenance(), "kernel": "default SDPA (flash / mem-efficient enabled)", "train_batches": {}}
    for seed, epochs in ((0, 10), (1, 5), (2, 5)):
        rng = np.random.default_rng(seed); bad = 0
        for _ in range(epochs):
            perm = rng.permutation(len(Xtr))
            bad += count_bad(model, Xtr, [perm[s:s + 8] for s in range(0, len(perm), 8)], dev)
        out["train_batches"]["seed%d_%d_epochs" % (seed, epochs)] = bad
    for k, X in (("val", Xva), ("test", Xte), ("fresh", Xf)):
        order = sorted(range(len(X)), key=lambda i: len(X[i]["ids"]))
        out[k] = count_bad(model, X, [order[s:s + 32] for s in range(0, len(order), 32)], dev)
    out["total_nonfinite_rows"] = sum(out["train_batches"].values()) + out["val"] + out["test"] + out["fresh"]
    print(out["train_batches"], out["val"], out["test"], out["fresh"], "total", out["total_nonfinite_rows"], flush=True)
    json.dump(out, open(os.path.join(R, "nan_scan_emotion.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
