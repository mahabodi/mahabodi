"""Fair opponent for experience memory: Laya itself trained on the SAME labelled examples.

Laya's decision head (type embedding + 2 transformer layers + scorer; laya/common.py DecisionModel)
is fine-tuned with cross-entropy on the same TRAIN examples MahaBodi's experience memory uses
(tune_experience.py `suites`: train[seed2][:2000]; prompt_injections train[seed2][100:]). The
encoder is frozen (DecisionModel supports detach_encoder=True): its token states are computed once
for every train/val/test row. Epoch is chosen on the same VALIDATION rows as the experience tuning;
the chosen model is evaluated ONCE on the bench.json test items. Full fine-tuning (encoder too) was
not run: on this CPU it would take many hours per task; this is stated in the results.

    .venv/bin/python research/finetune_laya_head.py --out research/results/laya_head_finetuned.json

v2 (reviewer: v1's emotion optimum sat on the grid corner, lr 1e-3 x epoch 6): lr grid 5e-5..3e-3,
extended upward x3 (up to 3 times) while the best lr is the largest; up to --epochs (30) per lr with
early stopping on validation (patience 5). Whether the chosen (lr, epoch) is on a grid boundary is
recorded. Encoder states are cached in research/cache/ft_enc_<suite>.pt. The chosen head is scored
ONCE on bench.json's test items 0..499 and ONCE on the third fresh sample (positions 1000..1499,
bench_fresh3_experience.json), where it is compared item by item with MahaBodi's default and
calibrate=200 arms. v1 (3 lrs x 6 epochs) is kept in laya_head_finetuned_v1_grid3x6.json.
Label: "Laya, head-only fine-tune (encoder frozen)"; full fine-tuning was not run.

v2 attempt 4 (written before running): on CUDA, PyTorch 2.2's fused SDPA kernels returned NaN hidden states
for some padded rows (boolq, prompt_injections encodings). The math SDP kernel is now forced on GPU, every
encoding is checked for non-finite values (counts recorded), and the hardware anchor reports NaN rows
separately from prediction differences (nan_rows, differ_excluding_nan).
Plateau tie-break (added after the ag_news head re-run picked a different setting on a +1-item validation
plateau; applies to selections made from now on, e.g. the clean-validation runs): among settings whose
validation accuracy equals the maximum, prefer the smallest learning rate, then the fewest epochs. Zero-shot
Laya counts as lr 0 / epoch 0, so a tie with zero-shot keeps zero-shot.
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
            dev = next(agent.model.parameters()).device
            h = agent.model.encoder(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).last_hidden_state
            for r, i in enumerate(order[s:s + bs]):
                L = len(items[i]["ids"])
                out[i] = (h[r, :L].to(torch.float16).cpu().clone(), items[i]["markers"], items[i]["qtype"])
    return out


def head_logits(dm, batch):
    dev = next(dm.parameters()).device
    hs = [x[0].float() for x in batch]
    L = max(h.shape[0] for h in hs); d = hs[0].shape[1]
    H = torch.zeros(len(batch), L, d); att = torch.zeros(len(batch), L, dtype=torch.long)
    K = max(len(x[1]) for x in batch)
    mpos = torch.zeros(len(batch), K, dtype=torch.long); mmask = torch.zeros(len(batch), K, dtype=torch.bool)
    for r, (h, mk, _) in enumerate(zip(hs, [x[1] for x in batch], [x[2] for x in batch])):
        H[r, :h.shape[0]] = h; att[r, :h.shape[0]] = 1
        mpos[r, :len(mk)] = torch.tensor(mk); mmask[r, :len(mk)] = True
    H, att, mpos, mmask = H.to(dev), att.to(dev), mpos.to(dev), mmask.to(dev)
    qt = torch.tensor([x[2] for x in batch], device=dev)
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lrs", default="5e-5,2e-4,1e-3,3e-3", help="starting learning-rate grid; (lr, epoch) chosen on validation")
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (Laya encoding and head training)")
    ap.add_argument("--val-source", default="standard", choices=["standard", "clean"],
                    help="clean: for ag_news/boolq use a validation set outside Laya's training mix (research/clean_val.py)")
    ap.add_argument("--save-head", default="", help="directory: save the chosen head state_dict per suite (head_<suite>.pt)")
    ap.add_argument("--out", default=os.path.join(R, "laya_head_finetuned.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    if a.device == "cuda":  # attempt 4: fused SDPA kernels give NaN on padded rows on this GPU
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    only = set(filter(None, a.only.split(",")))
    base = json.load(open(os.path.join(R, "bench.json")))["suites"]
    exp = json.load(open(os.path.join(R, "bench_experience.json")))["suites"]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"suites": {}}
    f3 = json.load(open(os.path.join(R, "bench_fresh3_experience.json")))["suites"]
    fresh = build_suites(1500, only or set(base))
    res.update({"laya_commit": LAYA_COMMIT, "label": "Laya, head-only fine-tune (encoder frozen)",
                "method": "head-only fine-tune (encoder frozen), cross-entropy, AdamW; lr grid %s extended x3 while the best lr is the largest; "
                          "<= %d epochs per lr, early stopping patience %d; (lr, epoch) chosen on validation" % (a.lrs, a.epochs, a.patience),
                "not_run": "full fine-tuning (encoder unfrozen): too slow on this CPU"})
    agent = Agent("convaiinnovations/laya", device=a.device)
    from provenance import provenance
    res["provenance"] = provenance(); res["device"] = a.device
    tr = train_suites(2000, 300)
    tests = build_suites(500, only or set(base))
    for name in (only or list(base)):
        if name in res["suites"] or name not in tests:
            continue
        T, S = tr[name], tests[name]
        qid, qd = next(iter(T["q"].items()))
        t0 = time.time()
        ytr = [label_idx(qd, T["y"](r)) for r in T["mem"]]
        yva = [label_idx(qd, T["y"](r)) for r in T["val"]]
        val_info = None
        if a.val_source == "clean":
            from clean_val import clean_val
            cv_states, yva, val_info = clean_val(name, [T["st"](r) for r in T["mem"]])
        F = fresh[name]
        if name in f3:
            assert F["gold"][1000:1500] == f3[name]["gold"], "fresh3 items differ"
        cache = os.path.join(ROOT, "research", "cache", "ft_enc_%s.pt" % name)
        if val_info is not None:
            Xtr, _, Xte, Xf = torch.load(cache) if os.path.exists(cache) else (encode_rows(agent, [T["st"](r) for r in T["mem"]], qd), None,
                                                                              encode_rows(agent, S["states"], S["q"]),
                                                                              encode_rows(agent, F["states"][1000:1500], S["q"]) if name in f3 else [])
            Xva = encode_rows(agent, cv_states, qd)
        elif os.path.exists(cache):
            Xtr, Xva, Xte, Xf = torch.load(cache)
        else:
            Xtr = encode_rows(agent, [T["st"](r) for r in T["mem"]], qd)
            Xva = encode_rows(agent, [T["st"](r) for r in T["val"]], qd)
            Xte = encode_rows(agent, S["states"], S["q"])
            Xf = encode_rows(agent, F["states"][1000:1500], S["q"]) if name in f3 else []
            torch.save((Xtr, Xva, Xte, Xf), cache)
        enc_s = time.time() - t0
        nonfinite = {k: sum(1 for x in X if not torch.isfinite(x[0].float()).all()) for k, X in (("train", Xtr), ("val", Xva), ("test", Xte), ("fresh", Xf))}
        print(name, "non-finite encoding rows", nonfinite, flush=True)
        # hardware anchor: zero-shot (epoch 0) argmax from THESE encodings vs the saved macOS laya_torch
        # predictions on the same items; any difference is hardware / encoding drift, not training
        nan_te = [not torch.isfinite(x[0].float()).all() for x in Xte]
        zp, _ = accuracy(agent.model, Xte, S["gold"])
        anchor = {"test_items_differ": sum(x != y for x, y in zip(zp, base[name]["laya_torch_pred"])), "test_n": len(zp), "test_nan_rows": sum(nan_te),
                  "test_differ_excluding_nan": sum(x != y for x, y, bad in zip(zp, base[name]["laya_torch_pred"], nan_te) if not bad)}
        if Xf:
            nan_f = [not torch.isfinite(x[0].float()).all() for x in Xf]
            zf, _ = accuracy(agent.model, Xf, f3[name]["gold"])
            anchor.update({"fresh3_items_differ": sum(x != y for x, y in zip(zf, f3[name]["laya_torch"]["pred"])), "fresh3_n": len(zf), "fresh3_nan_rows": sum(nan_f),
                           "fresh3_differ_excluding_nan": sum(x != y for x, y, bad in zip(zf, f3[name]["laya_torch"]["pred"], nan_f) if not bad)})
        anchor["flag_over_1pct"] = any(anchor.get(k + "_items_differ", 0) > 0.01 * anchor.get(k + "_n", 1) for k in ("test", "fresh3"))
        print(name, "anchor", anchor, flush=True)
        best, curve = None, []
        lrs = [float(x) for x in a.lrs.split(",")]
        li, ext = 0, 0
        while li < len(lrs):
            lr = lrs[li]; li += 1
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
            best_here, since = -1.0, 0
            for ep in range(1, a.epochs + 1):
                dm.train()
                perm = rng.permutation(len(Xtr))
                for s0 in range(0, len(perm), 16):
                    bi = perm[s0:s0 + 16]
                    lg = head_logits(dm, [Xtr[i] for i in bi])
                    loss = torch.nn.functional.cross_entropy(lg, torch.tensor([ytr[i] for i in bi], device=lg.device))
                    opt.zero_grad(); loss.backward(); opt.step()
                _, va = accuracy(dm, Xva, yva)
                curve.append({"lr": lr, "epoch": ep, "val_accuracy": round(va, 4)})
                print(name, "lr", lr, "epoch", ep, "val", round(va, 4), flush=True)
                if va > best[0] or (va == best[0] and (lr, ep) < (best[2], best[1])):  # plateau tie-break
                    best = (va, ep, lr, copy.deepcopy({k: v for k, v in dm.state_dict().items() if not k.startswith("encoder.")}))
                if va > best_here:
                    best_here, since = va, 0
                else:
                    since += 1
                    if since >= a.patience:
                        break
            if li == len(lrs) and best[2] == lrs[-1] and ext < 3:
                lrs.append(lrs[-1] * 3); ext += 1
        dm = copy.deepcopy(agent.model)
        if best[3] is not None:
            dm.load_state_dict(best[3], strict=False)
        if a.save_head:
            os.makedirs(a.save_head, exist_ok=True)
            hp = os.path.join(a.save_head, "head_%s.pt" % name)
            torch.save({k: v.cpu() for k, v in (best[3] or {}).items()}, hp)  # empty = zero-shot Laya chosen (epoch 0)
        pred, acc = accuracy(dm, Xte, S["gold"])
        fr = None
        if Xf:
            fg = f3[name]["gold"]
            fp, facc = accuracy(dm, Xf, fg)
            fc = [p == g for p, g in zip(fp, fg)]
            arm = lambda k: [p == g for p, g in zip(f3[name][k]["pred"], fg)]
            fr = {"items": "fresh3 positions 1000..1499", "accuracy": round(facc, 4), "ci95": wilson(sum(fc), len(fc)),
                  "laya_zero_shot_accuracy": f3[name]["laya_torch"]["accuracy"],
                  "mcnemar_vs_laya_zero_shot": mcnemar(fc, arm("laya_torch")),
                  "bodi_default_accuracy": f3[name]["gated_agree"]["accuracy"], "mcnemar_bodi_default_vs_finetuned": mcnemar(arm("gated_agree"), fc),
                  "bodi_calibrated_accuracy": f3[name]["calibrated"]["accuracy"], "mcnemar_bodi_calibrated_vs_finetuned": mcnemar(arm("calibrated"), fc),
                  "pred": fp}
        corr = [p == g for p, g in zip(pred, S["gold"])]
        lp = base[name]["laya_torch_pred"]
        e = exp[name]["bodi_experience_per_suite"]
        tried = sorted({c["lr"] for c in curve if c["lr"] > 0})
        r = {"val_source": a.val_source, "clean_val": val_info, "device": a.device, "sdp_kernel": "math" if a.device == "cuda" else "cpu", "nonfinite_encoding_rows": nonfinite, "hardware_anchor": anchor, "train": len(Xtr), "val": len(Xva), "chosen_epoch": best[1], "chosen_lr": best[2], "lrs_tried": tried,
             "lr_on_grid_boundary": best[2] in (tried[0], tried[-1]) if best[2] > 0 else None,
             "epoch_on_grid_boundary": best[1] == a.epochs, "fresh3": fr,
             "test_items": "bench.json 0..499 (MahaBodi per-suite experience settings were tuned for, and tested on, these items)",
             "val_curve": curve, "encode_seconds": round(enc_s, 1),
             "test_accuracy": round(acc, 4), "test_ci95": wilson(sum(corr), len(corr)),
             "mcnemar_vs_laya_zero_shot": mcnemar(corr, [p == g for p, g in zip(lp, S["gold"])]),
             "mcnemar_bodi_experience_vs_finetuned": mcnemar([p == g for p, g in zip(e["pred"], S["gold"])], corr),
             "bodi_experience_accuracy": e["accuracy"], "laya_zero_shot_accuracy": base[name]["laya_torch"]["accuracy"], "pred": pred}
        res["suites"][name] = r
        print(name, {k: v for k, v in r.items() if k not in ("pred", "val_curve")}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
