"""Fair opponent for experience memory: Laya itself trained on the SAME labelled examples.

Laya's decision head (type embedding + 2 transformer layers + scorer; laya/common.py DecisionModel)
is fine-tuned with cross-entropy on the same TRAIN examples MahaBodi's experience memory uses
(tune_experience.py `suites`: train[seed2][:2000]; prompt_injections train[seed2][100:]). The
encoder is frozen (DecisionModel supports detach_encoder=True): its token states are computed once
for every train/val/test row. Epoch is chosen on the same VALIDATION rows as the experience tuning;
the chosen model is evaluated ONCE on the bench.json test items. Full fine-tuning (encoder too) was
not run: on this CPU it would take many hours per task; this is stated in the results.

    .venv/bin/python research/finetune_laya_head.py --out research/results/laya_head_finetuned.json
"""
import argparse, copy, json, os, sys, time
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import LAYA_COMMIT, build_suites, mcnemar, wilson  # noqa: E402
from tune_experience import suites as train_suites, label_idx  # noqa: E402
from laya import Agent  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")


def encode_rows(agent, states, q, bs=16):
    """Frozen encoder states (fp16, CPU) per row, plus markers/qtype."""
    qi = agent._to_internal(q)
    items = []
    for st in states:
        ids, mk = build_sequence(agent.tok, st, qi, agent.cfg["max_len"], agent.cfg["head_max_len"], truncate_left=isinstance(st, list))
        items.append({"ids": ids, "markers": mk, "qtype": QTYPES[qi["t"]]})
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            sel = [items[i] for i in order[s:s + bs]]
            b = collate_items([sel], agent.tok.pad_token_id)
            h = agent.model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            for r, i in enumerate(order[s:s + bs]):
                L = len(items[i]["ids"])
                out[i] = (h[r, :L].to(torch.float16).clone(), items[i]["markers"], items[i]["qtype"])
    return out


def head_logits(dm, batch):
    hs = [x[0].float() for x in batch]
    L = max(h.shape[0] for h in hs); d = hs[0].shape[1]
    H = torch.zeros(len(batch), L, d); att = torch.zeros(len(batch), L, dtype=torch.long)
    K = max(len(x[1]) for x in batch)
    mpos = torch.zeros(len(batch), K, dtype=torch.long); mmask = torch.zeros(len(batch), K, dtype=torch.bool)
    for r, (h, mk, _) in enumerate(zip(hs, [x[1] for x in batch], [x[2] for x in batch])):
        H[r, :h.shape[0]] = h; att[r, :h.shape[0]] = 1
        mpos[r, :len(mk)] = torch.tensor(mk); mmask[r, :len(mk)] = True
    qt = torch.tensor([x[2] for x in batch])
    h = H + dm.type_emb(qt)[:, None, :]
    pad = att == 0
    for layer in dm.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    idx = mpos[:, :, None].expand(-1, -1, h.size(-1))
    logits = dm.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
    return logits.masked_fill(~mmask, -1e4)


def accuracy(dm, rows, gold, bs=32):
    dm.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            preds += head_logits(dm, rows[s:s + bs]).argmax(-1).tolist()
    return preds, float(np.mean([p == g for p, g in zip(preds, gold)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lrs", default="5e-5,2e-4,1e-3", help="learning-rate grid; (lr, epoch) chosen on validation")
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(R, "laya_head_finetuned.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    only = set(filter(None, a.only.split(",")))
    base = json.load(open(os.path.join(R, "bench.json")))["suites"]
    exp = json.load(open(os.path.join(R, "bench_experience.json")))["suites"]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"suites": {}}
    res.update({"laya_commit": LAYA_COMMIT, "method": "head-only fine-tune (encoder frozen), cross-entropy, AdamW, (lr in %s, epoch <= %d) chosen on validation" % (a.lrs, a.epochs),
                "not_run": "full fine-tuning (encoder unfrozen): too slow on this CPU"})
    agent = Agent("convaiinnovations/laya", device="cpu")
    tr = train_suites(2000, 300)
    tests = build_suites(500, only or set(base))
    for name in (only or list(base)):
        if name in res["suites"] or name not in tests:
            continue
        T, S = tr[name], tests[name]
        qid, qd = next(iter(T["q"].items()))
        t0 = time.time()
        Xtr = encode_rows(agent, [T["st"](r) for r in T["mem"]], qd)
        ytr = [label_idx(qd, T["y"](r)) for r in T["mem"]]
        Xva = encode_rows(agent, [T["st"](r) for r in T["val"]], qd)
        yva = [label_idx(qd, T["y"](r)) for r in T["val"]]
        Xte = encode_rows(agent, S["states"], S["q"])
        enc_s = time.time() - t0
        best, curve = None, []
        for lr in [float(x) for x in a.lrs.split(",")]:
            dm = copy.deepcopy(agent.model)
            for p in dm.encoder.parameters():
                p.requires_grad = False
            params = [p for n, p in dm.named_parameters() if not n.startswith("encoder.")]
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
            if best is None:
                _, va0 = accuracy(dm, Xva, yva)
                best = (va0, 0, 0.0, None)
                curve.append({"lr": 0.0, "epoch": 0, "val_accuracy": round(va0, 4)})
            rng = np.random.default_rng(0)
            for ep in range(1, a.epochs + 1):
                dm.train()
                perm = rng.permutation(len(Xtr))
                for s0 in range(0, len(perm), 16):
                    bi = perm[s0:s0 + 16]
                    loss = torch.nn.functional.cross_entropy(head_logits(dm, [Xtr[i] for i in bi]), torch.tensor([ytr[i] for i in bi]))
                    opt.zero_grad(); loss.backward(); opt.step()
                _, va = accuracy(dm, Xva, yva)
                curve.append({"lr": lr, "epoch": ep, "val_accuracy": round(va, 4)})
                print(name, "lr", lr, "epoch", ep, "val", round(va, 4), flush=True)
                if va > best[0]:
                    best = (va, ep, lr, copy.deepcopy({k: v for k, v in dm.state_dict().items() if not k.startswith("encoder.")}))
        dm = copy.deepcopy(agent.model)
        if best[3] is not None:
            dm.load_state_dict(best[3], strict=False)
        pred, acc = accuracy(dm, Xte, S["gold"])
        corr = [p == g for p, g in zip(pred, S["gold"])]
        lp = base[name]["laya_torch_pred"]
        e = exp[name]["bodi_experience_per_suite"]
        r = {"train": len(Xtr), "val": len(Xva), "chosen_epoch": best[1], "chosen_lr": best[2], "val_curve": curve, "encode_seconds": round(enc_s, 1),
             "test_accuracy": round(acc, 4), "test_ci95": wilson(sum(corr), len(corr)),
             "mcnemar_vs_laya_zero_shot": mcnemar(corr, [p == g for p, g in zip(lp, S["gold"])]),
             "mcnemar_bodi_experience_vs_finetuned": mcnemar([p == g for p, g in zip(e["pred"], S["gold"])], corr),
             "bodi_experience_accuracy": e["accuracy"], "laya_zero_shot_accuracy": base[name]["laya_torch"]["accuracy"], "pred": pred}
        res["suites"][name] = r
        print(name, {k: v for k, v in r.items() if k not in ("pred", "val_curve")}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
