"""Export sentence-transformers/all-MiniLM-L6-v2 to ONNX for MahaBodi's dense embedder.

Graph: (input_ids, attention_mask, token_type_ids) -> last_hidden_state. MahaBodi mean-pools
over the attention mask and L2-normalises in Rust, which is exactly the sentence-transformers
pipeline for this model (Transformer -> Pooling(mean) -> Normalize). Verified here against
SentenceTransformer.encode on real sentences of different lengths (cosine >= 0.9999), and a
golden file is written for the Rust parity test.

    .venv/bin/python research/export_embedder.py --out models/minilm
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    tok = AutoTokenizer.from_pretrained(a.repo)
    base = AutoModel.from_pretrained(a.repo).eval()

    class LastHidden(torch.nn.Module):  # drop BERT's pooler output: only the token states are needed
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state

    model = LastHidden(base).eval()
    os.makedirs(a.out, exist_ok=True)
    enc = tok(["hello world", "a longer sentence to make the batch ragged"], padding=True, return_tensors="pt")
    path = os.path.join(a.out, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]), path,
                          input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["last_hidden_state"],
                          dynamic_axes={k: {0: "batch", 1: "seq"} for k in ["input_ids", "attention_mask", "token_type_ids", "last_hidden_state"]},
                          opset_version=17, do_constant_folding=True)
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    st = SentenceTransformer(a.repo, device="cpu")
    texts = ["I was charged twice for the same order.", "hi", "The Lakers beat the Celtics 110-102 in overtime on Sunday night at home.",
             "报销需要收据", "What is the capital of France?"]
    ref = st.encode(texts, normalize_embeddings=True)
    e = tok(texts, padding=True, truncation=True, max_length=st.max_seq_length, return_tensors="np")
    h, = s.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64),
                      "token_type_ids": e["token_type_ids"].astype(np.int64)})
    m = e["attention_mask"][..., None].astype(np.float32)
    v = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cos = (v * ref).sum(1)
    print("onnx+meanpool vs SentenceTransformer cosine:", np.round(cos, 6).tolist())
    if cos.min() < 0.9999:
        sys.exit("embedder verification failed")
    snap = snapshot_download(a.repo, allow_patterns=["tokenizer.json", "tokenizer_config.json", "sentence_bert_config.json"])
    for f in ["tokenizer.json", "tokenizer_config.json"]:
        shutil.copy(os.path.join(snap, f), os.path.join(a.out, f))
    json.dump({"max_seq_length": st.max_seq_length, "pooling": "mean", "normalize": True, "source": a.repo},
              open(os.path.join(a.out, "embedder_config.json"), "w"), indent=1)
    json.dump({"texts": texts, "vectors": ref.tolist()}, open(os.path.join(a.out, "golden.json"), "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
