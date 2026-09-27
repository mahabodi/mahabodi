"""Cross-version check for the CLM encoder recipe (research/PREREG_CLM.md): the vLLM parity check ran the transformers side
under transformers 5.17.0 (vLLM venv), while the CLM encoder server runs under the main venv's 4.57.3. This embeds the
same 109 texts (from clm_vllm_recipe_check.json) with Qwen3-0.6B, CPU fp32, same recipe, under the running interpreter's
transformers, and saves the vectors; `--compare a.npy b.npy` reports per-text cosines. Pass: min cosine >= 0.999.

    <venv>/bin/python research/clm_transformers_version_check.py --embed out.npy
    python research/clm_transformers_version_check.py --compare a.npy b.npy --out research/results/clm_transformers_version_check.json
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = "Qwen/Qwen3-0.6B"
REV = "c1899de289a04d12100db370d81485cdf75e47ca"


def embed(out):
    import torch, transformers
    from transformers import AutoModel, AutoTokenizer
    texts = [x["text"] for x in json.load(open(os.path.join(HERE, "results", "clm_vllm_recipe_check.json")))["per_text"]]
    tok = AutoTokenizer.from_pretrained(M, revision=REV)
    m = AutoModel.from_pretrained(M, revision=REV, torch_dtype=torch.float32).eval()
    V = []
    with torch.inference_mode():
        for t in texts:
            ids = tok(t, add_special_tokens=True)["input_ids"]
            h = m(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].float()
            V.append((h / h.norm()).numpy())
    np.save(out, np.stack(V))
    json.dump({"transformers": transformers.__version__, "torch": torch.__version__, "n": len(texts)}, open(out + ".json", "w"))
    print("saved", out, transformers.__version__)


def compare(a, b, out):
    A, B = np.load(a), np.load(b)
    cos = (A * B).sum(1)
    res = {"model": M, "model_revision": REV, "a": json.load(open(a + ".json")), "b": json.load(open(b + ".json")),
           "n": int(len(cos)), "min_cos": float(cos.min()), "mean_cos": float(cos.mean()), "threshold": 0.999,
           "pass": bool(cos.min() >= 0.999), "per_text_cos": [float(c) for c in cos]}
    json.dump(res, open(out, "w"), indent=1)
    print({k: res[k] for k in ("a", "b", "n", "min_cos", "pass")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed"); ap.add_argument("--compare", nargs=2); ap.add_argument("--out")
    a = ap.parse_args()
    embed(a.embed) if a.embed else compare(*a.compare, a.out)
