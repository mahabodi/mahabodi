"""CPU stand-in for CLM's encoder server (research/PREREG_CLM.md): an OpenAI-compatible /v1/embeddings endpoint serving
Qwen3-8B with last-token pooling, the way `vllm serve Qwen/Qwen3-8B --runner pooling` does, so clm.Engine can run
unchanged on a machine whose GPU (11 GB) cannot hold the 8B encoder.

Recipe (vLLM pooling defaults for a causal LM):
- tokenize with the model's tokenizer (Qwen3 adds no special tokens);
- `truncate_prompt_tokens=k` keeps the LAST k tokens;
- the final hidden state (after the model's final RMSNorm) at the last token;
- L2-normalised (clm/embedder.py normalises again, which is idempotent).
fp32 by default (unquantised); --dtype bfloat16 to match vLLM's default precision exactly.
Encoder parity is checked end to end against CLM's documented README outputs before any scoring.

    .venv/bin/python research/clm_embed_server.py --model <Qwen3-8B dir> --port 8090 --threads 16
"""
import argparse, base64, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Encoder:
    def __init__(self, path, dtype, threads):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(path)
        t0 = time.time()
        self.model = AutoModel.from_pretrained(path, torch_dtype=getattr(torch, dtype)).eval()
        self.load_s = time.time() - t0
        self.lock = threading.Lock()

    @torch.inference_mode()
    def embed(self, texts, truncate):
        vecs, ntok = [], 0
        for t in texts:  # one at a time: no padding, exactly one sequence per forward like vLLM's per-request pooling
            ids = self.tok(t, add_special_tokens=True)["input_ids"]
            if truncate and len(ids) > truncate:
                ids = ids[-truncate:]
            ntok += len(ids)
            with self.lock:
                h = self.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].float()
            v = h / (h.norm() + 1e-12)
            vecs.append(v.numpy().astype(np.float32))
        return vecs, ntok


def make_handler(enc, served_name):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/embeddings":
                self.send_response(404); self.end_headers(); return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            vecs, ntok = enc.embed(inp, body.get("truncate_prompt_tokens"))
            b64 = body.get("encoding_format") == "base64"
            data = [{"index": i, "object": "embedding",
                     "embedding": base64.b64encode(v.tobytes()).decode() if b64 else v.tolist()} for i, v in enumerate(vecs)]
            out = json.dumps({"object": "list", "model": served_name, "data": data,
                              "usage": {"prompt_tokens": ntok, "total_tokens": ntok}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-name", default="qwen3-8b")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    enc = Encoder(a.model, a.dtype, a.threads)
    print("loaded %s (%s) in %.1fs" % (a.model, a.dtype, enc.load_s), flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(enc, a.served_name)).serve_forever()


if __name__ == "__main__":
    main()
