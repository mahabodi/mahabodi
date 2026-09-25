"""Strongest fair Laya baseline: Laya FULLY fine-tuned (encoder unfrozen) on the SAME labelled examples
MahaBodi stores as memory (pre-registered with the reviewer before running; complements the
head-only baseline in finetune_laya_head.py).

Same splits as finetune_laya_head.py: train = tune_experience.suites train[seed2][:2000]
(prompt_injections: train[seed2][100:]), validation = the same 300 (100) validation items, test =
bench.json items 0..499, fresh = bench_fresh3_experience.json items 1000..1499 (scored against the
SAVED per-item predictions of Laya zero-shot and MahaBodi).

Training: the whole DecisionModel (encoder + typed head), cross-entropy over the option markers, AdamW
(weight decay 0.01), linear warm-up over the first 10% of epoch 1, fp16 autocast + GradScaler, encoder
gradient checkpointing, batch 8. lr grid {1e-5, 2e-5, 5e-5}, extended x2.5 upward (or /2.5 downward)
up to 2 times while the best lr is the largest (smallest); <= 10 epochs per lr, early stopping on
validation (patience 3). (lr, epoch) chosen on validation, boundary flags recorded; the chosen model is
scored once on test and once on fresh. Hardware anchor: zero-shot argmax of this run's model vs the
saved macOS laya_torch predictions. Machine provenance recorded.

Added before running (reviewer):
  Seeds: the run of record is seed 0. The CHOSEN (lr, epoch) is then retrained with seeds 1 and 2
  and the fresh accuracy range over the 3 seeds is reported; if MahaBodi's verdict vs ft changes
  across seeds for a suite, that suite is "seed-dependent".
  Verdict rule: per suite, exact McNemar on fresh items, beat/loss at p < 0.05 else tie, for MahaBodi
  default and calibrated vs ft seed 0. Every loss is reported prominently.
  Cost: GPU wall-clock and peak GPU memory of training per suite, next to MahaBodi's learn() time on
  the same 2,000 examples on this machine's CPU.
  Failures: a suite that OOMs or diverges (non-finite loss) is recorded as "not run: <reason>", never
  dropped silently; no per-suite hyperparameter change without writing it down first.

Attempt 1 (laya_full_finetuned_attempt1_oom.json) OOMed on every suite; fixes written down BEFORE
attempt 2, implementation only (no hyperparameter change):
  (a) Laya's base model stays on the CPU; only the training copy (and a temporary copy for zero-shot
      predictions) is on the GPU. Pre-declared fallback if batch 8 still OOMs: batch 4 with gradient
      accumulation 2 (same effective batch 8); if that OOMs too, batch 2 with accumulation 4; if that OOMs,
      "not run: OOM at batch 2 on a 2080 Ti 11 GB" for that suite. No other change (no 8-bit optimisers, no
      shorter max_len) without a new written pre-registration. Recorded per suite: micro_batch, accumulation.

Attempt 2 (laya_full_finetuned.json): emotion trained (run of record, fp16); its seeds 1-2 OOMed because the
selected model stayed on the GPU during the seed retrains (bug); banking77, sst5, boolq, ag_news and
prompt_injections diverged on the first step in fp16 (non-finite loss). Attempt 3, written down before running:
  (a) --fp32: training without fp16 autocast (fp32 weights, grads and activations), same grid, patience,
      micro-batch fallback chain, for the 5 suites that diverged; recorded per suite as "precision".
  (b) seed fix: the selected model is scored, then freed from the GPU before seeds 1-2 are retrained;
      --seeds-only SUITE reruns just seeds 1-2 at that suite's chosen (lr, epoch) from the existing result,
      with the same precision as its run of record.

Attempt 3 (fp32) failed identically: the cause was NOT fp16. On CUDA, PyTorch 2.2's fused SDPA kernels return
NaN hidden states for some padded rows (the same batches are clean on CPU). Attempt 4, written down before
running: the math SDP kernel is forced on GPU for all encoding / training / evaluation; the original
pre-registered fp16 autocast is used again (the fp32 switch rested on the wrong diagnosis; fp32 only if
non-finite values recur under the math kernel); the hardware anchor reports NaN rows separately
(nan_rows, differ_excluding_nan). Emotion (attempt 2, default kernel) stays the run of record only if a
scan finds 0 non-finite logits over its train / val / test / fresh batches under that kernel
(research/scan_nan_emotion.py -> nan_scan_emotion.json).
  (b) All evaluation / prediction (anchor, validation selection, test, fresh) runs in fp32; fp16
      autocast is used for training only.

    .venv/bin/python research/finetune_laya_full.py --device cuda --out research/results/laya_full_finetuned.json
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
from provenance import provenance  # noqa: E402
from tune_experience import suites as train_suites, label_idx  # noqa: E402
from laya import Agent  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results")


def items_for(agent, states, q):
    qi = agent._to_internal(q)
    out = []
    for st in states:
        ids, mk = build_sequence(agent.tok, st, qi, agent.cfg["max_len"], agent.cfg["head_max_len"], truncate_left=isinstance(st, list))
        out.append({"ids": ids, "markers": mk, "qtype": QTYPES[qi["t"]]})
    return out


def logits_of(model, batch, dev, amp=False):
    b = collate_items([batch], 0)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev.type == "cuda"):
        lg, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev), b["marker_mask"].to(dev), b["qtype"].to(dev))
    return lg.float()


def predict(model, items, dev, bs=32, nan_flags=None):
    model.eval(); out = []
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    res = [None] * len(items)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            lg = logits_of(model, [items[i] for i in sel], dev)
            p = lg.argmax(-1).tolist(); bad = (~torch.isfinite(lg).all(-1)).tolist()
            for i, v, b_ in zip(sel, p, bad):
                res[i] = v
                if nan_flags is not None:
                    nan_flags[i] = b_
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lrs", default="1e-5,2e-5,5e-5")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(R, "laya_full_finetuned.json"))
    ap.add_argument("--fp32", action="store_true", help="train without fp16 autocast (attempt 3a)")
    ap.add_argument("--val-source", default="standard", choices=["standard", "clean"],
                    help="clean: for ag_news/boolq use a validation set outside Laya's training mix (research/clean_val.py)")
    ap.add_argument("--seeds-only", default="", help="suite: rerun only seeds 1-2 at its recorded chosen (lr, epoch) (attempt 3b)")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(a.device)
    if dev.type == "cuda":  # attempt 4: fused SDPA kernels give NaN on padded rows on this GPU
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    base = json.load(open(os.path.join(R, "bench.json")))["suites"]
    f3 = json.load(open(os.path.join(R, "bench_fresh3_experience.json")))["suites"]
    only = [x for x in a.only.split(",") if x] or ["emotion", "banking77", "sst5", "boolq", "ag_news", "prompt_injections"]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"suites": {}}
    res.update({"laya_commit": LAYA_COMMIT, "label": "Laya, FULL fine-tune (encoder + head)", "provenance": provenance(),
                "method": __doc__.split("\n\n")[2]})
    from mahabodi import Bodi
    bodi = Bodi(); bodi.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=8)
    bodi.load_embedder(os.path.join(ROOT, "models", "minilm"), intra_threads=8)
    agent = Agent("convaiinnovations/laya", device="cpu")  # base weights stay on the CPU (fix a)
    base_model = agent.model
    tr = train_suites(2000, 300)
    tests = build_suites(1500, set(only))
    if a.seeds_only:
        only = [a.seeds_only]
    for name in only:
        prev = res["suites"].get(name)
        if a.seeds_only:
            assert prev and "fresh3" in prev and prev["chosen_epoch"] > 0, "--seeds-only needs a trained result with fresh items"
        elif prev and "not_run" not in prev:
            continue
        T, S = tr[name], tests[name]
        qid, qd = next(iter(T["q"].items()))
        Xtr = items_for(agent, [T["st"](r) for r in T["mem"]], qd); ytr = [label_idx(qd, T["y"](r)) for r in T["mem"]]
        Xva = items_for(agent, [T["st"](r) for r in T["val"]], qd); yva = [label_idx(qd, T["y"](r)) for r in T["val"]]
        val_info = None
        if a.val_source == "clean":
            from clean_val import clean_val
            cv_states, yva, val_info = clean_val(name, [T["st"](r) for r in T["mem"]])
            Xva = items_for(agent, cv_states, qd)
        gold = S["gold"][:500]
        assert gold == base[name]["gold"]
        Xte = items_for(agent, S["states"][:500], S["q"])
        Xf = items_for(agent, S["states"][1000:1500], S["q"]) if name in f3 else []
        if Xf:
            assert S["gold"][1000:1500] == f3[name]["gold"]
        vr = lambda mc: ("beat" if mc["a_only"] > mc["b_only"] else "loss") if mc["p"] < 0.05 else "tie"

        def run_seeds(lr_, ep_):
            fg_ = f3[name]["gold"]; out_ = {}
            for sd in (1, 2):
                try:
                    ms = train(lr_, sd, fixed_epochs=ep_)
                    sp = predict(ms, Xf, dev); sc = [p == g for p, g in zip(sp, fg_)]
                    out_[str(sd)] = {"accuracy": round(float(np.mean(sc)), 4), "pred": sp,
                                "verdict_default_vs_ft": vr(mcnemar([p == g for p, g in zip(f3[name]["gated_agree"]["pred"], fg_)], sc))}
                    del ms
                except (FloatingPointError, torch.cuda.OutOfMemoryError) as e:
                    out_[str(sd)] = {"not_run": "%s: %s" % (type(e).__name__, e)}
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            return out_

        def summarize_seeds(fr_):
            accs = [v["accuracy"] for v in fr_["seeds"].values() if "accuracy" in v]
            fr_["seed_accuracy_range"] = [min(accs), max(accs)]
            fr_["seed_dependent"] = len({v["verdict_default_vs_ft"] for v in fr_["seeds"].values() if "verdict_default_vs_ft" in v}) > 1

        zs = copy.deepcopy(base_model).to(dev)  # temporary GPU copy for zero-shot predictions
        nt = [False] * len(Xte)
        z_te = predict(zs, Xte, dev, nan_flags=nt)
        anchor = {"test_items_differ": sum(x != y for x, y in zip(z_te, base[name]["laya_torch_pred"])), "test_n": len(z_te), "test_nan_rows": sum(nt),
                  "test_differ_excluding_nan": sum(x != y for x, y, b_ in zip(z_te, base[name]["laya_torch_pred"], nt) if not b_)}
        if Xf:
            nf = [False] * len(Xf)
            z_f = predict(zs, Xf, dev, nan_flags=nf)
            anchor.update({"fresh3_items_differ": sum(x != y for x, y in zip(z_f, f3[name]["laya_torch"]["pred"])), "fresh3_n": len(z_f), "fresh3_nan_rows": sum(nf),
                           "fresh3_differ_excluding_nan": sum(x != y for x, y, b_ in zip(z_f, f3[name]["laya_torch"]["pred"], nf) if not b_)})
        anchor["flag_over_1pct"] = any(anchor.get(k + "_items_differ", 0) > 0.01 * anchor.get(k + "_n", 1) for k in ("test", "fresh3"))
        print(name, "anchor", anchor, flush=True)
        va0 = float(np.mean([p == g for p, g in zip(predict(zs, Xva, dev), yva)]))
        del zs
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        best = (va0, 0, 0.0, None); curve = [{"lr": 0.0, "epoch": 0, "val_accuracy": round(va0, 4)}]
        lrs = [float(x) for x in a.lrs.split(",")]; li, ext_up, ext_dn = 0, 0, 0
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        micro = [a.batch]  # becomes a.batch // 2 (accumulation 2) after an OOM at batch a.batch (fallback a)

        def train(lr, seed, fixed_epochs=None, on_epoch=None):
            """Train from Laya's weights; early stopping unless fixed_epochs. Returns (model, epochs_run)."""
            torch.manual_seed(seed)
            model = copy.deepcopy(base_model).to(dev)
            model.encoder.gradient_checkpointing_enable()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            steps_ep = (len(Xtr) + a.batch - 1) // a.batch; warm = max(1, steps_ep // 10)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: min(1.0, (s_ + 1) / warm))
            scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda" and not a.fp32)
            rng = np.random.default_rng(seed); best_here, since = -1.0, 0
            for ep in range(1, (fixed_epochs or a.epochs) + 1):
                model.train()
                perm = rng.permutation(len(Xtr))
                for s0 in range(0, len(perm), a.batch):
                    bi = perm[s0:s0 + a.batch]
                    opt.zero_grad()
                    for m0 in range(0, len(bi), micro[0]):
                        mb = bi[m0:m0 + micro[0]]
                        lg = logits_of(model, [Xtr[i] for i in mb], dev, amp=not a.fp32)
                        loss = torch.nn.functional.cross_entropy(lg, torch.tensor([ytr[i] for i in mb], device=dev)) * len(mb) / len(bi)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite loss at lr %g seed %d epoch %d" % (lr, seed, ep))
                        scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update(); sched.step()
                if fixed_epochs:
                    continue
                va = float(np.mean([p == g for p, g in zip(predict(model, Xva, dev), yva)]))
                stop = on_epoch(model, lr, ep, va)
                if va > best_here:
                    best_here, since = va, 0
                else:
                    since += 1
                    if since >= a.patience or stop:
                        break
            return model

        def on_epoch(model, lr, ep, va):
            nonlocal best
            curve.append({"lr": lr, "epoch": ep, "val_accuracy": round(va, 4)})
            print(name, "lr", lr, "epoch", ep, "val", round(va, 4), round(time.time() - t0), flush=True)
            if va > best[0] or (va == best[0] and (lr, ep) < (best[2], best[1])):  # plateau tie-break
                best = (va, ep, lr, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            return False

        if a.seeds_only:  # attempt 3b: only seeds 1-2 at the recorded chosen setting
            prev["fresh3"]["seeds"].update(run_seeds(prev["chosen_lr"], prev["chosen_epoch"]))
            summarize_seeds(prev["fresh3"])
            prev["seeds_rerun"] = {"precision": "fp32" if a.fp32 else "fp16 autocast", "provenance": provenance()}
            print(name, "seeds", {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in prev["fresh3"]["seeds"].items()}, flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
            continue
        try:
            while li < len(lrs):
                lr = lrs[li]; li += 1
                try:
                    m_ = train(lr, 0, on_epoch=on_epoch)
                except torch.cuda.OutOfMemoryError:
                    if micro[0] <= 2:
                        raise torch.cuda.OutOfMemoryError("OOM at batch 2 on a 2080 Ti 11 GB")
                    torch.cuda.empty_cache(); micro[0] //= 2  # pre-declared fallbacks (a): 8 -> 4x2 -> 2x4
                    print(name, "OOM -> micro-batch", micro[0], "x accumulation", a.batch // micro[0], flush=True)
                    del curve[1:]; best = (va0, 0, 0.0, None); li = 0; lrs = [float(x) for x in a.lrs.split(",")]; ext_up = ext_dn = 0
                    continue
                del m_
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if li == len(lrs):
                    if best[2] == max(lrs) and ext_up < 2:
                        lrs.append(max(lrs) * 2.5); ext_up += 1
                    elif best[2] == min(lrs) and best[2] > 0 and ext_dn < 2:
                        lrs.append(min(lrs) / 2.5); ext_dn += 1
        except (FloatingPointError, torch.cuda.OutOfMemoryError) as e:
            res["suites"][name] = {"not_run": "%s: %s" % (type(e).__name__, e), "val_curve": curve, "hardware_anchor": anchor}
            print(name, "NOT RUN", e, flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            continue
        train_seconds = round(time.time() - t0, 1)
        peak_gb = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == "cuda" else None
        model = copy.deepcopy(base_model).to(dev)
        if best[3] is not None:
            model.load_state_dict(best[3])
        pred = predict(model, Xte, dev); corr = [p == g for p, g in zip(pred, gold)]
        tried = sorted({c["lr"] for c in curve if c["lr"] > 0})
        r = {"val_source": a.val_source, "clean_val": val_info, "device": a.device, "sdp_kernel": "math" if dev.type == "cuda" else "cpu", "micro_batch": micro[0], "accumulation": a.batch // micro[0], "hardware_anchor": anchor, "train": len(Xtr), "val": len(Xva), "chosen_lr": best[2], "chosen_epoch": best[1],
             "lrs_tried": tried, "lr_on_grid_boundary": (best[2] in (tried[0], tried[-1])) if best[2] > 0 else None,
             "epoch_on_grid_boundary": best[1] == a.epochs, "val_curve": curve,
             "test_accuracy": round(float(np.mean(corr)), 4), "test_ci95": wilson(sum(corr), len(corr)), "test_pred": pred,
             "mcnemar_test_vs_laya_zero_shot": mcnemar(corr, [p == g for p, g in zip(base[name]["laya_torch_pred"], gold)])}
        if Xf:
            fg = f3[name]["gold"]; fp = predict(model, Xf, dev); fc = [p == g for p, g in zip(fp, fg)]
            arm = lambda k: [p == g for p, g in zip(f3[name][k]["pred"], fg)]
            r["fresh3"] = {"items": "fresh3 positions 1000..1499", "accuracy": round(float(np.mean(fc)), 4), "ci95": wilson(sum(fc), len(fc)),
                           "laya_zero_shot_accuracy": f3[name]["laya_torch"]["accuracy"], "mcnemar_vs_laya_zero_shot": mcnemar(fc, arm("laya_torch")),
                           "bodi_default_accuracy": f3[name]["gated_agree"]["accuracy"], "mcnemar_bodi_default_vs_finetuned": mcnemar(arm("gated_agree"), fc),
                           "bodi_calibrated_accuracy": f3[name]["calibrated"]["accuracy"], "mcnemar_bodi_calibrated_vs_finetuned": mcnemar(arm("calibrated"), fc),
                           "knn_own_accuracy": f3[name]["knn_own"]["accuracy"], "pred": fp}
        vr = lambda mc: ("beat" if mc["a_only"] > mc["b_only"] else "loss") if mc["p"] < 0.05 else "tie"
        if Xf:
            fr = r["fresh3"]
            fr["verdict_default_vs_ft"] = vr(fr["mcnemar_bodi_default_vs_finetuned"])
            fr["verdict_calibrated_vs_ft"] = vr(fr["mcnemar_bodi_calibrated_vs_finetuned"])
            seeds = {"0": {"accuracy": fr["accuracy"], "verdict_default_vs_ft": fr["verdict_default_vs_ft"]}}
            del model  # fix 3b: free the selected model before the seed retrains
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if best[1] > 0:
                seeds.update(run_seeds(best[2], best[1]))
            fr["seeds"] = seeds
            summarize_seeds(fr)
        r["precision"] = "fp32" if a.fp32 else "fp16 autocast"
        r["train_gpu_seconds"] = train_seconds; r["peak_gpu_gb"] = peak_gb
        bodi.forget(); tl = time.time()
        bodi.learn([T["st"](x) for x in T["mem"]], T["q"], [{qid: T["y"](x)} for x in T["mem"]])
        r["mahabodi_learn_cpu_seconds"] = round(time.time() - tl, 1)  # same machine, CPU, 8 threads, default learn (no calibrate)
        r["seconds"] = round(time.time() - t0, 1)
        res["suites"][name] = r
        print(name, {k: v for k, v in r.items() if k not in ("val_curve", "test_pred", "fresh3")}, (r.get("fresh3") or {}).get("accuracy"), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
