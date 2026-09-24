"""Export a Laya checkpoint to ONNX for Bodi's native (Rust) System-1 runtime.

Laya's own exporter (laya-ts/scripts/export_onnx.py) needs torch.export dynamic shapes
(torch >= 2.5), which is not installable on x86_64 macOS (last wheel: torch 2.2.2). This
script uses the TorchScript exporter instead and writes ONE fused graph:

    (input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)

It verifies against the PyTorch model on REAL Laya-built sequences (padded batch of mixed
lengths and question types) before writing, and optionally writes an int8 dynamic-quantized
copy.

    .venv/bin/python research/export_onnx.py --out models/laya-v2 [--subfolder typed-decisions] [--int8]
"""
import argparse
import json
import math
import os
import shutil
import sys

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch


def act_rel_diff(a, b):
    """Relative act-head error max|a-b| / max|b|. Laya's act logits are O(1e3) (softmax is
    exactly [1, 0]), so a probability comparison would pass even with errors in the hundreds."""
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


class ShapeFreeLayer(torch.nn.Module):
    """Numerically identical, trace-safe forward for Laya's head layers
    (nn.TransformerEncoderLayer, norm_first=True, relu, eval mode).

    F.multi_head_attention_forward does Python arithmetic on sizes (bsz * num_heads), which the
    TorchScript tracer freezes into Reshape constants, so the exported head only ran at the
    traced (batch, seq). This re-expresses the same math with reshapes that never name batch
    or seq, reusing the layer's own weights."""

    def __init__(self, layer):
        super().__init__()
        # Fail loudly on any checkpoint whose head differs from the math re-expressed below.
        assert layer.norm_first, "ShapeFreeLayer assumes norm_first=True"
        assert layer.self_attn.batch_first, "ShapeFreeLayer assumes batch_first=True"
        assert layer.self_attn._qkv_same_embed_dim and layer.self_attn.in_proj_bias is not None
        assert layer.activation is torch.nn.functional.relu, "ShapeFreeLayer assumes relu activation"
        self.l = layer
        self.h = layer.self_attn.num_heads

    def forward(self, x, pad):
        l = self.l
        y = l.norm1(x)
        q, k, v = torch.nn.functional.linear(y, l.self_attn.in_proj_weight, l.self_attn.in_proj_bias).chunk(3, -1)
        d = q.size(-1) // self.h
        q, k, v = (t.unflatten(-1, (self.h, -1)).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(pad[:, None, None, :], float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).flatten(-2)
        x = x + l.self_attn.out_proj(o)
        x = x + l.linear2(torch.relu(l.linear1(l.norm2(x))))
        return x


class Fused(torch.nn.Module):
    """DecisionModel.forward (laya/common.py) with the head layers swapped for ShapeFreeLayer."""

    def __init__(self, dm):
        super().__init__()
        self.dm = dm
        self.layers = torch.nn.ModuleList(ShapeFreeLayer(l) for l in dm.head.layers) if dm.head is not None else None

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        dm = self.dm
        h = dm.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + dm.type_emb(qtype)[:, None, :]
        if self.layers is not None:
            pad = attention_mask == 0
            for layer in self.layers:
                h = layer(h, pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = dm.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits, -1)
        kk = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(kk)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, kk / 255.0], -1)
        act_logits = dm.act_head(torch.cat([h[:, 0].float(), feats], -1))
        # `pooled`: the [CLS] state after the decision head (question- and state-conditioned).
        # Extra output for MahaBodi's experience memory; logits/act_logits are unchanged.
        return logits, act_logits, h[:, 0].float()


def sample_batch(agent):
    from laya.common import collate_items, QTYPES
    cases = [
        ({"article": "The Lakers beat the Celtics 110-102 in overtime on Sunday."},
         {"t": "choice", "ins": "What is the topic of `article`?",
          "crit": {"world": "world news", "sports": "sports", "business": "business",
                   "sci_tech": "science and technology"}}),
        ("I was charged twice for the same order, please refund me " * 20,
         {"t": "noul", "ins": "Does the customer ask for a refund?", "crit": None}),
        ({"review": "It was fine I guess, nothing special."},
         {"t": "score", "ins": "How positive is `review`?",
          "crit": ["very negative", "negative", "neutral", "positive", "very positive"]}),
    ]
    from laya.common import build_sequence
    items = []
    for st, q in cases:
        seq, markers = build_sequence(agent.tok, st, q, agent.cfg.get("max_len", 512),
                                      agent.cfg.get("head_max_len", 192))
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    return collate_items([items], agent.tok.pad_token_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--int8", action="store_true", help="also write model.int8.onnx (dynamic quantization)")
    a = ap.parse_args()

    from laya import Agent
    agent = Agent(a.repo, subfolder=a.subfolder, device="cpu")
    dm = agent.model.float().eval()
    fused = Fused(dm).eval()
    b = sample_batch(agent)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    with torch.inference_mode():
        ref_logits, ref_act = dm(*args)          # Laya's own forward is the reference
        mine = fused(*args)[0]
    d0 = float((mine - ref_logits).abs().max())
    print("ShapeFreeLayer vs Laya forward: max|dlogit|=%.2e" % d0)
    if d0 > 1e-4:
        sys.exit("ShapeFreeLayer does not reproduce Laya's head")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "model.onnx")
    # nn.TransformerEncoderLayer takes a fused "fast path" (aten::_transformer_encoder_layer_fwd,
    # no ONNX symbolic) whenever grad is disabled. torch 2.2 has no switch for it, but the fast
    # path is skipped while grad is enabled and params require grad, so trace in that mode.
    with torch.enable_grad():
        torch.onnx.export(
            fused, args, path,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act_logits", "pooled"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                "marker_pos": {0: "batch", 1: "markers"}, "marker_mask": {0: "batch", 1: "markers"},
                "qtype": {0: "batch"}, "logits": {0: "batch", 1: "markers"}, "act_logits": {0: "batch"}, "pooled": {0: "batch"},
            },
            opset_version=a.opset, do_constant_folding=True,
        )

    import onnxruntime as ort
    feeds = {"input_ids": args[0].numpy(), "attention_mask": args[1].numpy(),
             "marker_pos": args[2].numpy(), "marker_mask": args[3].numpy(), "qtype": args[4].numpy()}

    def check(p, tol, strict=True):
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        lg, ac = s.run(["logits", "act_logits"], feeds)
        mask = feeds["marker_mask"]
        d = float(np.abs(np.where(mask, lg - ref_logits.numpy(), 0)).max())
        da = float(np.abs(ac - ref_act.numpy()).max())
        dp = act_rel_diff(ac, ref_act.numpy())
        same_argmax = bool((np.where(mask, lg, -1e9).argmax(-1) ==
                            np.where(mask, ref_logits.numpy(), -1e9).argmax(-1)).all())
        print("%s: max|dlogit|=%.2e max|dact_logit|=%.2e rel|dact|=%.2e argmax_equal=%s"
              % (os.path.basename(p), d, da, dp, same_argmax))
        if strict and (d > tol or dp > 1e-5 or not same_argmax):
            sys.exit("verification failed for %s" % p)

    check(path, 1e-3)
    check_dynamic(path, agent, dm)

    if a.int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = os.path.join(a.out, "model.int8.onnx")
        quantize_dynamic(path, q, weight_type=QuantType.QInt8)
        # int8 is NOT verified here: diffs are printed for the record only. Whether it keeps
        # accuracy is decided by the benchmark (research/bench.py), never by this sanity batch.
        check(q, 0.0, strict=False)

    snap = agent_dir(a.repo, a.subfolder)
    shutil.copy(os.path.join(snap, "tokenizer", "tokenizer.json"), os.path.join(a.out, "tokenizer.json"))
    shutil.copy(os.path.join(snap, "rl_agent_config.json"), os.path.join(a.out, "rl_agent_config.json"))
    tc = os.path.join(snap, "tokenizer", "tokenizer_config.json")
    if os.path.exists(tc):
        shutil.copy(tc, os.path.join(a.out, "tokenizer_config.json"))
    # Golden fixture for Rust tokenizer/sequence parity tests.
    with open(os.path.join(a.out, "golden.json"), "w") as f:
        json.dump({"input_ids": feeds["input_ids"].tolist(), "attention_mask": feeds["attention_mask"].tolist(),
                   "marker_pos": feeds["marker_pos"].tolist(), "marker_mask": feeds["marker_mask"].tolist(),
                   "qtype": feeds["qtype"].tolist(), "logits": ref_logits.numpy().tolist()}, f)
    print("wrote", a.out)


def check_dynamic(path, agent, dm):
    """The TorchScript exporter can bake traced sizes into the graph as constants. Re-verify at
    batch/seq/marker shapes different from the trace: batch 1 short, batch 5 near max_len with
    mixed option counts (2..12)."""
    import onnxruntime as ort
    from laya.common import build_sequence, collate_items, QTYPES
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    long_text = " ".join("The quarterly report shows revenue growth of %d percent in region %d." % (i, i)
                         for i in range(80))
    batches = [
        [("hi", {"t": "noul", "ins": "Is this a greeting?", "crit": None})],
        [(long_text, {"t": "choice", "ins": "Pick the label", "crit": {"opt%d" % i: None for i in range(k)}})
         for k in (2, 5, 8, 12)] + [(long_text, {"t": "score", "ins": "Rate it", "crit": ["low", "mid", "high"]})],
    ]
    for cases in batches:
        items = []
        for st, q in cases:
            seq, markers = build_sequence(agent.tok, st, q, 512, 192)
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        b = collate_items([items], agent.tok.pad_token_id)
        args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        with torch.inference_mode():
            ref, ref_a = dm(*args)
        lg, ac = s.run(["logits", "act_logits"], {"input_ids": args[0].numpy(), "attention_mask": args[1].numpy(),
                             "marker_pos": args[2].numpy(), "marker_mask": args[3].numpy(),
                             "qtype": args[4].numpy()})
        m = b["marker_mask"].numpy()
        d = float(np.abs(np.where(m, lg - ref.numpy(), 0)).max())
        da = float(np.abs(ac - ref_a.numpy()).max())
        dp = act_rel_diff(ac, ref_a.numpy())
        print("dynamic check batch=%d seq=%d markers=%d: max|dlogit|=%.2e max|dact_logit|=%.2e rel|dact|=%.2e"
              % (args[0].shape[0], args[0].shape[1], args[2].shape[1], d, da, dp))
        if d > 1e-3 or dp > 1e-5:
            sys.exit("dynamic-shape verification failed")


def agent_dir(repo, subfolder):
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo, allow_patterns=[(subfolder + "/" if subfolder else "") + "*.json",
                                                (subfolder + "/" if subfolder else "") + "tokenizer/*"])
    return os.path.join(d, subfolder) if subfolder else d


if __name__ == "__main__":
    main()
