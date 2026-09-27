"""Dev-only: retrieval recall over the full 5.9M pages for candidate query forms (research/PREREG_MILLION_SCALE.md: every
setting tuned on dev). Query forms: the mention string alone; the mention +-50 characters of context; the mention +-200
(the current state). For dense (MiniLM, exact over the shared index) and BM25 (shared index). Test mentions are not
touched.

    .venv/bin/python research/el_query_forms.py
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402

K = "/media/sda/data/kilt"


def window(inp, c):
    a, b = inp.find("[START_ENT]"), inp.find("[END_ENT]")
    if a < 0 or b < 0:
        return inp[:2 * c]
    return inp[max(0, a - c):min(len(inp), b + len("[END_ENT]") + c)].strip()


def main():
    dev_ids = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(K, "el", "mentions.jsonl")) if json.loads(l)["split"] == "dev"}
    raw = {}
    for line in open(os.path.join(K, "aidayago2-train-kilt.jsonl")):
        d = json.loads(line)
        if d["id"] in dev_ids:
            raw[d["id"]] = d["input"]
    ms = list(dev_ids.values())
    forms = {"mention": [m["mention"] or "" for m in ms],
             "ctx50": [window(raw[m["id"]], 50) for m in ms],
             "ctx200": [m["state"] for m in ms]}
    gold = np.array([m["gold_row"] for m in ms])
    import torch
    from sentence_transformers import SentenceTransformer
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev_); st.max_seq_length = 256
    E = np.load(os.path.join(K, "dense", "emb.f16.npy"), mmap_mode="r")
    import bm25s, Stemmer
    bm = bm25s.BM25.load(os.path.join(K, "bm25"))
    out = {"n_dev": len(ms), "provenance": provenance(), "forms": {}}
    for name, qs in forms.items():
        Q = torch.tensor(st.encode(qs, batch_size=256, normalize_embeddings=True), device=dev_, dtype=torch.float16)
        bs = torch.full((len(qs), 100), -1e4, device=dev_); bi = torch.zeros((len(qs), 100), dtype=torch.long, device=dev_)
        for s in range(0, E.shape[0], 500_000):
            blk = torch.tensor(np.asarray(E[s:s + 500_000]), device=dev_, dtype=torch.float16)
            v, i = torch.topk((Q @ blk.T).float(), 100, dim=1)
            cs, ci = torch.cat([bs, v], 1), torch.cat([bi, i + s], 1)
            bs, j = torch.topk(cs, 100, dim=1); bi = torch.gather(ci, 1, j)
        dtop = bi.cpu().numpy()
        tok = bm25s.tokenize(qs, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)
        btop, _ = bm.retrieve(tok, k=100, show_progress=False, n_threads=8)
        rec = lambda top: {"@%d" % k: round(float(np.mean([g in t[:k] for g, t in zip(gold, top)])), 4) for k in (1, 10, 100)}
        out["forms"][name] = {"dense": rec(dtop), "bm25": rec(btop),
                              "either@100": round(float(np.mean([g in d or g in b for g, d, b in zip(gold, dtop, btop)])), 4)}
        print(name, out["forms"][name], flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "el_query_forms_dev.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
